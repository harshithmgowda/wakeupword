import os

path = '/root/openwakeword-training/.venv/lib/python3.10/site-packages/torch_audiomentations/utils/io.py'
if os.path.exists(path):
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    content = content.replace('torchaudio.USE_SOUNDFILE_LEGACY_INTERFACE = False', '# torchaudio legacy flag removed')
    content = content.replace('torchaudio.set_audio_backend("soundfile")', '# torchaudio backend call removed')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print('Successfully patched torch_audiomentations/utils/io.py')
else:
    print('File not found:', path)

