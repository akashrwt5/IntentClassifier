import os
import re
import time
import socket
import pandas as pd
from deep_translator import GoogleTranslator

# Set global socket timeout to prevent indefinite hangs
socket.setdefaulttimeout(15)

# Input and output file paths
input_file = "/Users/akashrawat/PycharmProjects/IntentClassifier/datasets/04_GENERATED_MASTER_training_data.csv"
output_file = "/Users/akashrawat/PycharmProjects/IntentClassifier/datasets/Generated_Master_training_French_Data.csv"

# Protected terms regex mappings
replacements = [
    (r'\b[Ee]dge\s+[Mm]ode\b', '__EDGEMODE__'),
    (r'\b[Tt]hrive\s+[Ss]core\b', '__THRIVESCORE__'),
    (r'\b[Ii]ntelli[Vv]oice\b', '__INTELLIVOICE__'),
    (r'\b[Ss]elf\s+[Cc]heck\b', '__SELFCHECK__'),
    (r'\b[Hh]ear[Ss]hare\b', '__HEARSHARE__'),
    (r'\b[Hh]earing\s+[Cc]are\s+[Aa]nywhere\b', '__HCA__'),
    (r'\b[Bb]rain\s+[Ss]core\b', '__BRAINSCORE__'),
    (r'\b[Bb]ody\s+[Ss]core\b', '__BODYSCORE__'),
    (r'\b[Tt]hrive\s+[Aa]ssistant\b', '__THRIVEASSISTANT__')
]

def pre_process_text(text):
    if not isinstance(text, str):
        return text
    for pattern, placeholder in replacements:
        text = re.sub(pattern, placeholder, text)
    return text

def post_process_text(text):
    if not isinstance(text, str):
        return text
    # Restore placeholders to their official French terminology
    text = text.replace('__EDGEMODE__', 'mode Edge')
    text = text.replace('__THRIVESCORE__', 'score Thrive')
    text = text.replace('__INTELLIVOICE__', 'IntelliVoice')
    text = text.replace('__SELFCHECK__', 'Self Check')
    text = text.replace('__HEARSHARE__', 'HearShare')
    text = text.replace('__HCA__', 'Hearing Care Anywhere')
    text = text.replace('__BRAINSCORE__', 'score cérébral')
    text = text.replace('__BODYSCORE__', 'score corporel')
    text = text.replace('__THRIVEASSISTANT__', 'assistant Thrive')
    
    # Fix generic "aids" translation errors to ensure NLU accuracy
    text = re.sub(r'\bmes aides\b(?!\s+auditives)', 'mes aides auditives', text, flags=re.IGNORECASE)
    text = re.sub(r'\bmon aide\b(?!\s+auditive)', 'mon aide auditive', text, flags=re.IGNORECASE)
    text = re.sub(r'\bmes appareils\b(?!\s+auditifs)', 'mes appareils auditifs', text, flags=re.IGNORECASE)
    text = re.sub(r'\bmon appareil\b(?!\s+auditif)', 'mon appareil auditif', text, flags=re.IGNORECASE)
    text = re.sub(r'\bdes aides\b(?!\s+auditives)', 'des aides auditives', text, flags=re.IGNORECASE)
    text = re.sub(r'\bde mes aides\b(?!\s+auditives)', 'de mes aides auditives', text, flags=re.IGNORECASE)
    text = re.sub(r'\bd\'aides\b(?!\s+auditives)', "d'aides auditives", text, flags=re.IGNORECASE)
    text = re.sub(r'\bdes appareils\b(?!\s+auditifs)', 'des appareils auditifs', text, flags=re.IGNORECASE)
    text = re.sub(r'\bd\'appareils\b(?!\s+auditifs)', "d'appareils auditifs", text, flags=re.IGNORECASE)
    
    # Clean up literal translation of aids to 'sida'
    text = re.sub(r'\bsida\b', 'aides auditives', text, flags=re.IGNORECASE)
    
    return text

def main():
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        return
        
    print(f"Loading {input_file}...")
    df = pd.read_csv(input_file)
    total_rows = len(df)
    print(f"Loaded {total_rows} rows.")
    
    # Pre-process text column (lowercase 'text')
    print("Applying terminology shielding placeholders...")
    df['text_shielded'] = df['text'].apply(pre_process_text)
    
    translator = GoogleTranslator(source='en', target='fr')
    
    # Load existing translations if output file exists to resume from checkpoint
    translated_clean = []
    if os.path.exists(output_file):
        try:
            df_existing = pd.read_csv(output_file)
            translated_clean = df_existing['text'].tolist()
            print(f"Resuming from checkpoint. Loaded {len(translated_clean)} existing translations.")
        except Exception as e:
            print(f"Failed to load checkpoint file ({e}), starting from scratch.")
            
    print("Starting translation in batches of 100...")
    
    batch_size = 100
    texts_to_translate = df['text_shielded'].tolist()
    
    for i in range(0, total_rows, batch_size):
        if i < len(translated_clean):
            # Already translated and checkpointed, skip
            continue
            
        batch = texts_to_translate[i : i + batch_size]
        
        # Translate the batch
        batch_translated_shielded = []
        for attempt in range(3):
            try:
                batch_translated_shielded = translator.translate_batch(batch)
                break
            except Exception as e:
                print(f"\nError translating batch starting at row {i} (attempt {attempt+1}): {e}")
                if attempt == 2:
                    # Fallback to appending original shielded text if all retries fail
                    batch_translated_shielded = batch
                else:
                    time.sleep(3) # Wait before retry
                    
        # Apply terminology restoring and cleaning for this batch
        batch_translated_clean = [post_process_text(t) for t in batch_translated_shielded]
        translated_clean.extend(batch_translated_clean)
        
        # Checkpoint: Save progress to output_file
        df_progress = pd.DataFrame({
            'text': translated_clean,
            'intent': df['intent'].iloc[:len(translated_clean)]
        })
        df_progress.to_csv(output_file, index=False)
        print(f"Progress: {len(translated_clean)}/{total_rows} rows translated and saved to checkpoint.")
        
        time.sleep(1.5) # Safe cooldown sleep to avoid rate limiting
        
    print(f"Successfully saved completed translated dataset to {output_file}!")

if __name__ == "__main__":
    main()
