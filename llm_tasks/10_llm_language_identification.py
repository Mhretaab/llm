from langdetect import detect

text = "I am learning how to use LangDetect for language identification."
text2 = "እታ ከተማ ኣዝያ ጽብቅቲ እያ። ብ በዝሒ በጻሕቲ ኸኣ ምልእ ዝበለት እያ።"
print(detect(text))
#print(detect(text2))
print(detect("Bonjour tout le monde"))  # French