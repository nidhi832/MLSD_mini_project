with open(r'C:\Users\admin\.gemini\antigravity\brain\8bdbb4cc-c3dc-4406-a012-8392af92514f\.system_generated\steps\272\content.md', 'r', encoding='utf-8') as f:
    text = f.read()

import re
matches = re.findall(r'https://[^\s"\'<>]+\.(?:png|jpg|jpeg|svg|webp)', text)
print('Total media URLs found:', len(matches))
for m in sorted(list(set(matches))):
    if 'airhelp' in m:
        print('AH:', m)

pos = text.find('One fee')
if pos != -1:
    print('\nFound One fee section:')
    print(text[max(0, pos-200):pos+600])

pos2 = text.find('NEWSLETTER')
if pos2 != -1:
    print('\nFound Newsletter section:')
    print(text[max(0, pos2-200):pos2+600])
