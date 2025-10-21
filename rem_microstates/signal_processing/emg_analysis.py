def evaluate_emg_twitching(epoch):
    emg_data = epoch.copy().pick_channels(['Menton']).get_data().squeeze()
    std = emg_data.std()
    return min(1.0, std / 20)  # Normalise, std > 20µV --> score proche de 1
