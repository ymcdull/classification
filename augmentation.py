from google_trans_new import google_translator

translator = google_translator()

sentence = ['stay hungry, stay foolish -- said by Steve Jobs', 'Apple stock gains while Facebook slumps']

translated_cn = translator.translate(sentence, lang_tgt='zh-cn')
print(translated_cn)

translated_en = translator.translate(translated_cn, lang_tgt='en')
print(translated_en)