import pathlib

data_dir = pathlib.Path('data/mini_speech_commands')
commands = [c.name for c in sorted(data_dir.iterdir()) if c.is_dir() and c.name != '_background_noise_']
print("Commands:", commands)
print("Number of commands:", len(commands))

# check one folder has actual wav files
sample_dir = data_dir / commands[0]
print(f"Sample files in '{commands[0]}':", list(sample_dir.glob('*.wav'))[:3])