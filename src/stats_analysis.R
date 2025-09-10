#!/usr/bin/env Rscript
# Shapiro (normalité), Student, Wilcoxon rank-sum sur la bandpower (abs/rel)
# Entrée : all_band_powers.csv (colonnes: base, group, abs_<Band>, rel_<Band>)
# Sorties: normality.csv, ttest.csv, wilcoxon.csv dans --out-dir

suppressPackageStartupMessages({
  if (!requireNamespace("optparse", quietly = TRUE)) {
    stop("Le package 'optparse' est requis. Installez-le avec install.packages('optparse').")
  }
  if (!requireNamespace("nortest", quietly = TRUE)) {
    stop("Le package 'nortest' est requis. Installez-le avec install.packages('nortest').")
  }
})

library(optparse)

BANDS_DEFAULT <- c("Delta","Theta","Alpha","Beta","Gamma")

# ---------- utils ----------
safe_shapiro <- function(x) {
  x <- x[is.finite(x)]
  n <- length(x)
  if (n < 3) return(list(W=NA_real_, p=NA_real_, n=n, method="too_few"))
  if (n <= 5000) {
    s <- shapiro.test(x)
    return(list(W=unname(s$statistic), p=s$p.value, n=n, method="shapiro"))
  } else {
    # Anderson–Darling comme fallback pour n>5000
    a <- nortest::ad.test(x)
    return(list(W=NA_real_, p=a$p.value, n=n, method="anderson-darling"))
  }
}

cohen_d_ind <- function(x, y) {
  x <- x[is.finite(x)]; y <- y[is.finite(y)]
  nx <- length(x); ny <- length(y)
  if (nx < 2 || ny < 2) return(NA_real_)
  sx <- stats::sd(x); sy <- stats::sd(y)
  sp <- sqrt(((nx-1)*sx^2 + (ny-1)*sy^2) / (nx+ny-2))
  if (!is.finite(sp) || sp == 0) return(NA_real_)
  (mean(x) - mean(y)) / sp
}

# Détecte si 'stat' est W (somme des rangs) ou déjà U, et calcule r in [-1,1]
# Disambiguation W vs U à partir de la p-value (approx normale)
rank_biserial_from_stat <- function(stat, n1, n2, p_raw = NA_real_) {
  Wmin <- n1 * (n1 + 1) / 2
  Wmax <- n1 * n2 + Wmin
  U_from_W <- stat - Wmin
  U_from_U <- stat

  # cas non ambigu : clairement U
  if (stat < Wmin || stat > Wmax) {
    U <- U_from_U
  } else if (is.finite(p_raw)) {
    # approx normale de la p à partir de U
    p_from_U <- function(U) {
      mu <- n1*n2/2
      sd <- sqrt(n1*n2*(n1+n2+1)/12)
      z <- (U - mu)/sd
      2*pnorm(-abs(z))
    }
    pW <- p_from_U(U_from_W)
    pU <- p_from_U(U_from_U)
    # prends l'interprétation la plus proche de p_raw
    U <- if (abs(pW - p_raw) <= abs(pU - p_raw)) U_from_W else U_from_U
  } else {
    # fallback sans p_raw : prends celle la plus éloignée du nul si la stat est extrême
    mu <- n1*n2/2
    U <- if (abs(U_from_U - mu) >= abs(U_from_W - mu)) U_from_U else U_from_W
  }

  r <- 2*U/(n1*n2) - 1
  max(min(r, 1), -1)
}


read_csv <- function(path) {
  utils::read.csv(path, stringsAsFactors = FALSE, check.names = FALSE)
}
write_csv <- function(df, path) {
  utils::write.csv(df, path, row.names = FALSE)
}

# ---------- CLI ----------
option_list <- list(
  make_option("--csv", type="character", default="/home/darryld/documents/EEG/preprocessed/bipolaire/3_results_analysis/all_band_powers.csv", help="Chemin vers all_band_powers.csv"),
  make_option("--out-dir", type="character", default="/home/darryld/documents/EEG/preprocessed/bipolaire/3_results_analysis", help="Dossier de sortie des CSV"),
  make_option("--metric", type="character", default="rel", help="abs | rel"),
  make_option("--bands", type="character", default=paste(BANDS_DEFAULT, collapse=",")),
  make_option("--exclude-unknown", action="store_true", default=FALSE, help="Exclure le groupe 'unknown'"),
  make_option("--min-n", type="integer", default=3, help="Taille min par groupe")
)
args <- parse_args(OptionParser(option_list=option_list))

csv_path <- args$csv
out_dir  <- args$`out-dir`
bands <- trimws(strsplit(args$bands, ",")[[1]])
metric <- match.arg(args$metric, c("abs","rel"))
dir.create(file.path(out_dir), recursive = TRUE, showWarnings = FALSE)

# ---------- data ----------
df <- read_csv(csv_path)
if (!all(c("base","group") %in% names(df))) stop("Colonnes 'base' et 'group' manquantes")
df$group <- tolower(trimws(df$group))
if (args$`exclude-unknown`) df <- subset(df, group != "unknown")

# Containers résultats
normal_rows <- list()
tt_rows <- list()
wx_rows <- list()

for (band in bands) {
  col <- paste0(metric, "_", band)
  if (!col %in% names(df)) {
    message("[WARN] colonne absente: ", col, " -> skip"); next
  }
  tmp <- df[, c("group", col)]
  names(tmp) <- c("group", "val")
  tmp <- tmp[is.finite(tmp$val), , drop = FALSE]

  # valeurs par groupe avec n >= min-n
  groups <- split(tmp$val, tmp$group)
  groups <- groups[vapply(groups, length, 1L) >= args$`min-n`]
  if (length(groups) < 2L) {
    message("[", col, "] <2 groupes valides -> skip"); next
  }

  # ---- Shapiro par groupe
  for (g in names(groups)) {
    res <- safe_shapiro(groups[[g]])
    normal_rows[[length(normal_rows)+1]] <- data.frame(
      metric = metric, band = band, group = g, n = res$n,
      shapiro_W = res$W, shapiro_p = res$p, method = res$method,
      stringsAsFactors = FALSE
    )
  }

  # ---- Comparaisons par paires
  labs <- sort(names(groups))
  pairs <- t(combn(labs, 2))

  # --- Student (var.equal = TRUE) ---
  p_t_raw <- numeric(nrow(pairs))
  t_stats  <- numeric(nrow(pairs))
  d_eff    <- numeric(nrow(pairs))
  n1s <- n2s <- numeric(nrow(pairs))

  for (i in seq_len(nrow(pairs))) {
    g1 <- pairs[i,1]; g2 <- pairs[i,2]
    x <- groups[[g1]]; y <- groups[[g2]]
    n1s[i] <- length(x); n2s[i] <- length(y)
    tt <- stats::t.test(x, y, var.equal = TRUE)  # Student "classique"
    p_t_raw[i] <- tt$p.value
    t_stats[i] <- unname(tt$statistic)
    d_eff[i]   <- cohen_d_ind(x, y)
  }
  p_t_holm <- p.adjust(p_t_raw, method = "holm")
  tt_df <- data.frame(
    metric = metric, band = band,
    group1 = pairs[,1], group2 = pairs[,2],
    t_stat = t_stats, p_raw = p_t_raw, p_holm = p_t_holm,
    cohen_d = d_eff, n1 = n1s, n2 = n2s, stringsAsFactors = FALSE
  )
  tt_rows[[length(tt_rows)+1]] <- tt_df

  # --- Wilcoxon rank-sum (Mann–Whitney), two-sided ---
  p_w_raw <- numeric(nrow(pairs))
  W_stats <- numeric(nrow(pairs))
  rb_eff  <- numeric(nrow(pairs))
  w_n1    <- numeric(nrow(pairs))
  w_n2    <- numeric(nrow(pairs))

  for (i in seq_len(nrow(pairs))) {
    g1 <- pairs[i,1]; g2 <- pairs[i,2]
    x <- groups[[g1]]; y <- groups[[g2]]
    nx <- length(x); ny <- length(y)
    w_n1[i] <- nx; w_n2[i] <- ny

    wx <- stats::wilcox.test(x, y, paired = FALSE, alternative = "two.sided",
                             exact = FALSE, correct = FALSE)
    stat <- unname(wx$statistic)
    W_stats[i] <- stat
    p_w_raw[i] <- wx$p.value
    rb_eff[i]  <- rank_biserial_from_stat(stat, nx, ny)
  }

  if (any(abs(rb_eff) > 1 + 1e-12)) {
    warning("rank_biserial hors [-1,1] : vérifie la conversion W/U")
  }

  p_w_holm <- p.adjust(p_w_raw, method = "holm")
  wx_df <- data.frame(
    metric = metric, band = band,
    group1 = pairs[,1], group2 = pairs[,2],
    W_stat = W_stats, p_raw = p_w_raw, p_holm = p_w_holm,
    rank_biserial = rb_eff, n1 = w_n1, n2 = w_n2,
    stringsAsFactors = FALSE
  )
  wx_rows[[length(wx_rows)+1]] <- wx_df
}

# ---------- export ----------
if (length(normal_rows)) write_csv(do.call(rbind, normal_rows), file.path(out_dir, "normality.csv"))
if (length(tt_rows))      write_csv(do.call(rbind, tt_rows),      file.path(out_dir, "ttest.csv"))
if (length(wx_rows))      write_csv(do.call(rbind, wx_rows),      file.path(out_dir, "wilcoxon.csv"))

cat(">>> Sauvegarde : ", file.path(out_dir, "normality.csv"), "\n")
cat(">>> Sauvegarde : ", file.path(out_dir, "ttest.csv"), "\n")
cat(">>> Sauvegarde : ", file.path(out_dir, "wilcoxon.csv"), "\n")
