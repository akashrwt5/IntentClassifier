import pandas as pd
from deep_translator import GoogleTranslator
import time
import os



file_name = "../data/pva_intent_new_one.csv"


if not os.path.exists(file_name):
    print(f"Error: '{file_name}' aapke is folder mein nahi mili!")
    print("Kripya check karein ki script aur CSV file dono ek hi folder mein hain.")
    exit()

# File load karein
df = pd.read_csv(file_name)
print(f"File successfully load ho gayi! Total rows: {len(df)}")

# 2. Translator setup
translator = GoogleTranslator(source='en', target='fr')

print("Translating... Please wait (Isme thoda waqt lagega kyunki data bada hai)...")

translated_texts = []
total_rows = len(df)

# 3. Loop ke sath batch translation aur safe error handling
for index, text in enumerate(df['Text']):
    if isinstance(text, str) and text.strip():
        # Retry mechanism agar internet temporary drop ho jaye
        for attempt in range(3):
            try:
                translated_text = translator.translate(text)
                translated_texts.append(translated_text)
                break
            except Exception as e:
                if attempt == 2:
                    print(f"\nRow {index} par dikkat aayi, original text hi chhod rahe hain.")
                    translated_texts.append(text)
                else:
                    time.sleep(2)  # Error aane par 2 second ruk kar fir try karein
    else:
        translated_texts.append(text)

    # Har 50 rows ke baad thoda break taaki Google block na kare
    if (index + 1) % 50 == 0:
        print(f"Progress: {index + 1}/{total_rows} rows complete...")
        time.sleep(1)  # 1 second ka pause system ko safe rakhne ke liye

# Column
df['Text'] = translated_texts

# 4. Final file save karein
output_file = "../data/pva_intent_french.csv"
df.to_csv(output_file, index=False)
print(f"\nDone! Aapki file '{output_file}' ke naam se successfully save ho gayi hai. 🎉")