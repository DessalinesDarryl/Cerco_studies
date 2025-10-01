#!/usr/bin/env Rscript

# ================== Bootstrap ==================
options(repos = c(CRAN = "https://cloud.r-project.org"))

suppressPackageStartupMessages({
  library(dplyr, warn.conflicts = FALSE)
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("Le package {ggplot2} est requis.")
  }
  library(ggplot2)
  if (!requireNamespace("rstatix", quietly = TRUE)) {
    stop("Le package {rstatix} est requis pour les post-hoc. Installe-le (conda-forge: r-rstatix).")
  }
})

# ================= Utils =================
safe_shapiro <- function(x) {
  x <- x[is.finite(x)]
  if (length(x) < 3) return(list(W=NA_real_, p=NA_real_))
  sh <- shapiro.test(x); list(W=unname(sh$statistic), p=unname(sh$p.value))
}

# Levene robuste (Brown–Forsythe) si {car} dispo, sinon fallback médiane
safe_levene <- function(y, g) {
  if (requireNamespace("car", quietly = TRUE)) {
    out <- try(car::leveneTest(y ~ g), silent = TRUE)
    if (!inherits(out, "try-error")) {
      return(list(statistic = unname(out$`F value`[1]),
                  p.value   = unname(out$`Pr(>F)`[1])))
    }
  }
  ok <- is.finite(y) & !is.na(g); y <- y[ok]; g <- g[ok]
  if (length(y) < 3 || length(unique(g)) < 2) {
    return(list(statistic = NA_real_, p.value = NA_real_))
  }
  med_by_g <- tapply(y, g, stats::median, na.rm = TRUE)
  z <- abs(y - med_by_g[as.character(g)])
  a <- aov(z ~ g); sm <- summary(a)
  list(statistic = unname(sm[[1]]$`F value`[1]),
       p.value   = unname(sm[[1]]$`Pr(>F)`[1]))
}

# Switchs 
RUN_POSTHOC   <- TRUE   # calcule les post-hoc (CSV)
DRAW_ANNOT    <- FALSE  # dessine (ou non) les annotations sur les plots

# ================= Groupes & palette =================
GROUP_LABEL_MAP <- c(
  "autoimmune encephalitides" = "Eai",
  "narcolepsy"                = "Narco",
  "synucleopathy"             = "Syn",
  "tcspi"                     = "Tcspi"
)

PALETTE <- c(
  "Eai"     = "#C9D175",
  "Narco"   = "#F15854",
  "Syn"     = "#44AA99",
  "Tcspi"   = "#BEBEBE",
  "unknown" = "#000000"
)

recode_groups <- function(df, group_col = "group") {
  if (!group_col %in% names(df)) return(df)
  g <- as.character(df[[group_col]])
  short <- ifelse(g %in% names(GROUP_LABEL_MAP), GROUP_LABEL_MAP[g], "unknown")
  lvls <- c("Eai","Narco","Syn","Tcspi")
  if (any(short == "unknown")) lvls <- c(lvls, "unknown")
  df[[group_col]] <- factor(short, levels = lvls)
  df
}

order_bands <- function(df, band_col = "band") {
  if (!band_col %in% names(df)) return(df)
  ord <- c("delta","theta","alpha","beta","gamma")
  if (is.character(df[[band_col]]) || is.factor(df[[band_col]])) {
    lvls <- intersect(ord, unique(as.character(df[[band_col]])))
    other <- setdiff(unique(as.character(df[[band_col]])), lvls)
    df[[band_col]] <- factor(df[[band_col]], levels = c(lvls, sort(other)))
  }
  df
}

# ============= Outliers IQR (PAR channel × group × band) =============
detect_outliers_channel_band <- function(df, k = 1.5, extreme_k = 3, na.rm = TRUE) {
  cols_expected <- c("base","group","channel","band","abs","rel")
  if (!all(cols_expected %in% names(df))) {
    stop("detect_outliers_channel_band(): colonnes manquantes: ",
         paste(setdiff(cols_expected, names(df)), collapse = ", "))
  }
  iqr_bounds <- function(x, na.rm = TRUE, k = 1.5) {
    x <- x[is.finite(x)]
    if (length(x) < 4) return(list(q1=NA_real_, q3=NA_real_, iqr=NA_real_, low=NA_real_, up=NA_real_))
    q1 <- unname(quantile(x, 0.25, na.rm=na.rm, names=FALSE))
    q3 <- unname(quantile(x, 0.75, na.rm=na.rm, names=FALSE))
    iqr <- q3 - q1
    list(q1=q1, q3=q3, iqr=iqr, low=q1 - k*iqr, up=q3 + k*iqr)
  }
  rows <- list(); k_row <- 0L
  chans <- sort(unique(df$channel))
  for (ch in chans) {
    sub_ch <- df[df$channel == ch, ]
    bands <- sort(unique(sub_ch$band))
    for (b in bands) {
      sub_cb <- sub_ch[sub_ch$band == b, ]
      groups <- sort(unique(sub_cb$group))
      for (g in groups) {
        sub <- sub_cb[sub_cb$group == g, ]
        for (metric in c("abs","rel")) {
          v <- sub[[metric]]
          bd_main <- iqr_bounds(v, na.rm=na.rm, k=k)
          bd_ext  <- iqr_bounds(v, na.rm=na.rm, k=extreme_k)
          if (is.na(bd_main$low) || is.na(bd_main$up)) next
          is_out <- (v < bd_main$low) | (v > bd_main$up)
          if (!any(is_out, na.rm = TRUE)) next
          is_ext <- (v < bd_ext$low) | (v > bd_ext$up)
          take <- which(is_out & is.finite(v)); if (!length(take)) next
          part <- sub[take, c("base","group","channel","band")]
          part$metric       <- metric
          part$value        <- v[take]
          part$lower_bound  <- bd_main$low
          part$upper_bound  <- bd_main$up
          part$is_outlier   <- TRUE
          part$is_extreme   <- is_ext[take]
          part$q1           <- bd_main$q1
          part$q3           <- bd_main$q3
          part$iqr          <- bd_main$iqr
          k_row <- k_row + 1L
          rows[[k_row]] <- part
        }
      }
    }
  }
  if (!length(rows)) {
    return(data.frame(
      base=character(), group=character(), channel=character(), band=character(),
      metric=character(), value=numeric(),
      lower_bound=numeric(), upper_bound=numeric(),
      is_outlier=logical(), is_extreme=logical(),
      q1=numeric(), q3=numeric(), iqr=numeric(),
      stringsAsFactors = FALSE
    ))
  }
  do.call(rbind, rows)
}

remove_iqr_outliers_channel_band <- function(df, metric = "abs", k = 3) {
  if (!metric %in% names(df)) return(df)
  bound_fun <- function(v, side = "low") {
    v <- v[is.finite(v)]; if (length(v) < 4) return(rep(NA_real_, length(v)))
    qs <- quantile(v, c(0.25, 0.75), na.rm = TRUE, names = FALSE)
    i  <- qs[2] - qs[1]
    lim <- if (side == "low") qs[1] - k*i else qs[2] + k*i
    rep(lim, length(v))
  }
  low <- ave(df[[metric]], df$channel, df$band, df$group, FUN = function(v) bound_fun(v, "low"))
  up  <- ave(df[[metric]], df$channel, df$band, df$group, FUN = function(v) bound_fun(v, "up"))
  keep <- is.na(low) | is.na(up) | (df[[metric]] >= low & df[[metric]] <= up)
  df[keep | !is.finite(df[[metric]]) | is.na(df[[metric]]), , drop = FALSE]
}

# ============= Post-hoc + annotations (robuste) =============
compute_posthoc_channel_band <- function(df) {
  if (!requireNamespace("rstatix", quietly = TRUE)) {
    stop("Le package {rstatix} est requis pour les post-hoc.")
  }
  req_cols <- c("channel","band","group","abs","rel")
  if (!all(req_cols %in% names(df))) {
    stop("Colonnes manquantes dans df: ", paste(setdiff(req_cols, names(df)), collapse=", "))
  }

  df <- as.data.frame(df)
  df$channel <- as.character(df$channel)
  df$band    <- as.character(df$band)
  df$group   <- droplevels(factor(df$group))

  post_rows <- list(); a_rows <- list(); k <- 0L
  chans <- sort(unique(df$channel))
  bands <- sort(unique(df$band))

  for (ch in chans) {
    sub_ch <- df[df$channel == ch, , drop = FALSE]
    for (b in bands) {
      sub_cb <- sub_ch[sub_ch$band == b, , drop = FALSE]
      if (!nrow(sub_cb)) next

      for (metric in c("abs","rel")) {
        if (!metric %in% names(sub_cb)) next
        dat <- if (metric == "abs") remove_iqr_outliers_channel_band(sub_cb, metric = "abs", k = 3) else sub_cb

        keep <- is.finite(dat[[metric]]) & !is.na(dat$group)
        dat  <- dat[keep, , drop = FALSE]
        if (!nrow(dat) || length(unique(dat$group)) < 2) next
        dat$group <- droplevels(factor(dat$group))

        grp_lvls <- levels(dat$group)
        if (is.null(grp_lvls) || length(grp_lvls) < 2) next

        sh_ps <- vapply(grp_lvls, function(g) {
          xg <- dat[dat$group == g, metric]
          if (sum(is.finite(xg)) < 3) return(NA_real_)
          safe_shapiro(xg)$p
        }, numeric(1))
        all_normal <- all(is.na(sh_ps) | sh_ps > 0.05)
        lev_p <- tryCatch(safe_levene(dat[[metric]], dat$group)$p.value, error = function(e) NA_real_)
        homo  <- !is.na(lev_p) && lev_p > 0.05

        ph <- tryCatch({
          if (all_normal && homo) {
            out <- rstatix::tukey_hsd(as.formula(paste(metric, "~ group")), data = dat)
            out$test <- "TukeyHSD"
            if (!"p.adj" %in% names(out)) out$p.adj <- out$p
            out$p_adj <- out$p.adj
            out
          } else {
            out <- rstatix::dunn_test(as.formula(paste(metric, "~ group")), data = dat,
                                      p.adjust.method = "bonferroni")
            out$test <- "Dunn-Bonf"
            if (!"p.adj" %in% names(out)) out$p.adj <- out$p
            out$p_adj <- out$p.adj
            out
          }
        }, error = function(e) NULL)

        if (is.null(ph) || !nrow(ph)) next
        needed <- c("group1","group2","p_adj","test")
        if (!all(needed %in% names(ph))) next

        k <- k + 1L
        pr <- data.frame(
          channel = ch, band = b, metric = metric, test = ph$test,
          group1 = ph$group1, group2 = ph$group2,
          p_raw = if ("p" %in% names(ph)) ph$p else NA_real_,
          p_adj = ph$p_adj,
          stars = ifelse(ph$p_adj <= 0.005, "***",
                  ifelse(ph$p_adj <= 0.01,  "**",
                  ifelse(ph$p_adj <= 0.05,  "*", ""))),
          stringsAsFactors = FALSE
        )
        post_rows[[k]] <- pr

        ann <- subset(pr, stars != "")
        if (nrow(ann)) {
          ymax <- max(dat[[metric]], na.rm = TRUE)
          ymin <- min(dat[[metric]], na.rm = TRUE)
          step <- max((ymax - ymin) * 0.08, ifelse(is.finite(ymax), ymax * 0.02, 0.1))
          ann$y.position <- ymax + step * seq_len(nrow(ann))
          a_rows[[length(a_rows) + 1L]] <- ann
        }
      }
    }
  }

  posthoc <- if (length(post_rows)) do.call(rbind, post_rows) else data.frame()
  annos   <- if (length(a_rows))   do.call(rbind, a_rows)   else data.frame()
  list(posthoc = posthoc, annos = annos)
}

# ============= Tests globaux (log) =============
run_tests_channel_band <- function(df) {
  rows <- list(); k <- 0L
  chans <- sort(unique(df$channel))
  bands <- levels(df$band); if (is.null(bands)) bands <- sort(unique(df$band))
  for (ch in chans) {
    sub_ch <- df[df$channel == ch, ]
    for (b in bands) {
      sub_cb <- sub_ch[sub_ch$band == b, ]
      if (!nrow(sub_cb)) next
      for (metric in c("abs","rel")) {
        dat <- if (metric == "abs") remove_iqr_outliers_channel_band(sub_cb, metric = "abs", k = 3) else sub_cb
        dat <- dat[is.finite(dat[[metric]]) & !is.na(dat$group), , drop = FALSE]
        if (!nrow(dat) || length(unique(dat$group)) < 2) next
        sh_ps <- sapply(levels(dat$group), function(g) {
          xg <- dat[dat$group == g, metric]
          if (sum(is.finite(xg)) < 3) return(NA_real_)
          safe_shapiro(xg)$p
        })
        all_normal <- all(is.na(sh_ps) | sh_ps > 0.05)
        lev <- safe_levene(dat[[metric]], dat$group)
        lev_p <- lev$p.value
        homo <- !is.na(lev_p) && lev_p > 0.05
        test_used <- NA_character_; stat <- NA_real_; pval <- NA_real_
        if (all_normal && homo) {
          a <- aov(dat[[metric]] ~ dat$group)
          sm <- summary(a)
          test_used <- "ANOVA"
          stat <- unname(sm[[1]]$`F value`[1]); pval <- unname(sm[[1]]$`Pr(>F)`[1])
        } else {
          kw <- kruskal.test(dat[[metric]] ~ dat$group)
          test_used <- "Kruskal-Wallis"; stat <- unname(kw$statistic); pval <- unname(kw$p.value)
        }
        k <- k + 1L
        rows[[k]] <- data.frame(channel=ch, band=b, metric=metric, test=test_used,
                                statistic=stat, p_value=pval,
                                stringsAsFactors = FALSE)
      }
    }
  }
  if (!length(rows)) return(data.frame())
  do.call(rbind, rows)
}

# ============== Boxplots par canal × bande (annotables) ==============
.safe_filename <- function(x) gsub("[^A-Za-z0-9_-]+", "-", x)

make_boxplots_per_channel_band <- function(df, out_dir, annos = NULL,
                                           hide_outliers = TRUE,
                                           palette = PALETTE) {
  if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  used_lvls <- levels(df$group)
  has_ggpubr <- requireNamespace("ggpubr", quietly = TRUE)

  chans <- sort(unique(df$channel))
  for (ch in chans) {
    sub_ch <- df[df$channel == ch, ]
    bands <- levels(sub_ch$band); if (is.null(bands)) bands <- sort(unique(sub_ch$band))
    for (b in bands) {
      sub_cb <- sub_ch[sub_ch$band == b, ]
      if (!nrow(sub_cb)) next
      sub_cb$group <- factor(sub_cb$group, levels = used_lvls)
      sub_cb_abs <- remove_iqr_outliers_channel_band(sub_cb, metric = "abs", k = 3)

      for (metric in c("abs","rel")) {
        dat <- if (metric == "abs") sub_cb_abs else sub_cb

        p <- ggplot(dat, aes(x = group, y = .data[[metric]], fill = group)) +
          geom_boxplot(outlier.shape = if (hide_outliers) NA else 16,
                       outlier.colour = if (hide_outliers) NA else "red",
                       outlier.size = if (hide_outliers) NA else 2) +
          labs(title = paste0("Channel ", ch, " - ", b, " - ", metric),
               x = "Group", y = metric) +
          scale_fill_manual(values = palette, drop = FALSE) +
          theme_minimal(base_size = 14) +
          theme(legend.position = "bottom",
                plot.margin = margin(10, 20, 10, 10)) +
          coord_cartesian(clip = "off")

        if (has_ggpubr && !is.null(annos) && nrow(annos)) {
          a <- subset(annos, channel == ch & band == b & metric == metric)
          if (nrow(a)) {
            p <- p + ggpubr::stat_pvalue_manual(a, label = "stars",
                                                xmin = "group1", xmax = "group2",
                                                y.position = "y.position",
                                                tip.length = 0.01, hide.ns = TRUE)
          }
        }

        fn <- file.path(out_dir, sprintf("boxplot_%s_%s_%s.png",
                                         .safe_filename(ch), .safe_filename(b), metric))
        ggsave(filename = fn, plot = p, width = 6, height = 5, dpi = 300)
      }
    }
  }
}

# ============== Grandes figures récap =================
make_boxplots_all_channels_bands <- function(df, out_dir, hide_outliers = TRUE, palette = PALETTE) {
  if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  df <- df %>%
    mutate(group = factor(group, levels = levels(group)),
           band  = if (is.factor(band)) band else factor(band),
           channel = if (is.factor(channel)) channel else factor(channel))

  for (metric in c("abs","rel")) {
    dat <- if (metric == "abs") remove_iqr_outliers_channel_band(df, metric = "abs", k = 3) else df

    p <- ggplot(dat, aes(x = group, y = .data[[metric]], fill = group)) +
      geom_boxplot(outlier.shape = if (hide_outliers) NA else 16) +
      scale_fill_manual(values = palette, drop = FALSE) +
      labs(title = paste0("All channels × bands — ", metric),
           x = "Group", y = metric) +
      theme_minimal(base_size = 13) +
      theme(legend.position = "bottom") +
      facet_grid(rows = vars(band), cols = vars(channel), scales = "free_y")

    fn <- file.path(out_dir, sprintf("boxgrid_all-channels_all-bands_%s.png", metric))
    ggsave(fn, p, width = 16, height = 10, dpi = 300)
  }
}

make_boxplots_per_band_all_channels <- function(df, out_dir, hide_outliers = TRUE, palette = PALETTE) {
  if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  bands <- levels(df$band); if (is.null(bands)) bands <- sort(unique(df$band))

  for (b in bands) {
    sub <- df[df$band == b, , drop = FALSE]
    if (!nrow(sub)) next
    sub$channel <- if (is.factor(sub$channel)) sub$channel else factor(sub$channel)

    for (metric in c("abs","rel")) {
      dat <- if (metric == "abs") remove_iqr_outliers_channel_band(sub, metric = "abs", k = 3) else sub

      p <- ggplot(dat, aes(x = group, y = .data[[metric]], fill = group)) +
        geom_boxplot(outlier.shape = if (hide_outliers) NA else 16) +
        scale_fill_manual(values = palette, drop = FALSE) +
        labs(title = paste0("Band ", b, " — ", metric),
             x = "Group", y = metric) +
        theme_minimal(base_size = 13) +
        theme(legend.position = "bottom") +
        facet_wrap(~ channel, scales = "free_y")

      fn <- file.path(out_dir, sprintf("boxwrap_band-%s_all-channels_%s.png",
                                       .safe_filename(b), metric))
      ggsave(fn, p, width = 14, height = 10, dpi = 300)
    }
  }
}

# =================== Pipeline ===================
per_channel_path <- "/home/darryld/documents/EEG/preprocessed/bipolaire/3_results_analysis/gp2/per_channel_band_powers.csv"

if (!file.exists(per_channel_path)) {
  stop("Fichier introuvable : ", per_channel_path)
}

df <- read.csv(per_channel_path, stringsAsFactors = FALSE)
df <- recode_groups(df, "group")
df <- order_bands(df, "band")

# Export outliers
outliers_cb <- detect_outliers_channel_band(df, k = 1.5, extreme_k = 3)
write.csv(outliers_cb,
          sub("per_channel_band_powers.csv","outliers_per_channel_band.csv",per_channel_path),
          row.names = FALSE)

# 1) Post-hoc + annotations
ph <- compute_posthoc_channel_band(df)
write.csv(ph$posthoc,
          sub("per_channel_band_powers.csv","posthoc_channel_band_pairs.csv", per_channel_path),
          row.names = FALSE)

# (optionnel) tests globaux
stats_cb <- run_tests_channel_band(df)
write.csv(stats_cb,
          sub("per_channel_band_powers.csv","stats_channel_band_anova_kruskal.csv",per_channel_path),
          row.names = FALSE)

# 2) Plots détaillés (canal × bande) avec annotations
plot_dir <- dirname(per_channel_path)
make_boxplots_per_channel_band(df, plot_dir, annos = ph$annos,
                               hide_outliers = TRUE, palette = PALETTE)

# 3) Grandes figures récap
make_boxplots_all_channels_bands(df, plot_dir, hide_outliers = TRUE, palette = PALETTE)
make_boxplots_per_band_all_channels(df, plot_dir, hide_outliers = TRUE, palette = PALETTE)

cat("OK: outliers + posthoc + ANOVA/KW + plots individuels + grandes figures\n")
