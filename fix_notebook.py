import json

path = r'c:\Users\User\Documents\Desktop\biztelai\app\notebooks\chat_eda_analysis.ipynb'

try:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for cell in data['cells']:
        if cell['cell_type'] == 'code' and 'outputs' not in cell:
            cell['outputs'] = []

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

    print("Notebook format fixed successfully!")
except Exception as e:
    print(f"Error: {e}")