import os
import json

# USERNAME and KAGGLE KEY must be filled with own values
kaggle_data = {
    "username": "[USERNAME]",
    "key": "[KAGGLE KEY]"
}

# .kaggle klasörünün yolunu belirle
kaggle_path = os.path.expanduser('~/.kaggle')

# Klasör yoksa oluştur
os.makedirs(kaggle_path, exist_ok=True)

# kaggle.json dosyasını doğru yere yaz
with open(os.path.join(kaggle_path, 'kaggle.json'), 'w') as f:
    json.dump(kaggle_data, f)

print("Kaggle token succesfull")