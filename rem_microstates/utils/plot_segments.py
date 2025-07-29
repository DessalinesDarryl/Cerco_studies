import matplotlib.pyplot as plt

def plot_microstates(windows, labels):
    for i, (win, label) in enumerate(zip(windows, labels)):
        plt.figure()
        win.plot(n_channels=6, title=f"{label.upper()} REM - Window {i}", show=True)
        plt.savefig(f"output/window_{i}_{label}.png")
