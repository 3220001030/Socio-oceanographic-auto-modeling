import pandas as pd
import json

# Load CSV file
csv_file = "/Users/gurumakaza/Library/CloudStorage/OneDrive-MacauUniversityofScienceandTechnology/D盘/【实验】文章/【文章】MCIGI/2024年02月04日元模型/Figures/Training.csv"  # Replace with your actual file path
df = pd.read_csv(csv_file)

# Convert each row to JSON structure
result = []
for _, row in df.iterrows():
    message_entry = {
        "conversations": [
            {"role": "system", "content": row["system"]},
            {"role": "user", "content": row["user"]},
            {"role": "assistant", "content": row["assistant"]}
        ],
        "text": [
            {"role": "system", "content": row["system"]},
            {"role": "user", "content": row["user"]},
            {"role": "assistant", "content": row["assistant"]}
        ]
    }
    result.append(message_entry)

# Convert list to JSON string
json_output = "\n".join(json.dumps(entry, ensure_ascii=False) for entry in result)

# Save to a JSON file (optional)
with open("output.json", "w", encoding="utf-8") as f:
    f.write(json_output)

# Print output
print(json_output)