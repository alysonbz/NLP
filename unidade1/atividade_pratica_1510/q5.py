import re

def achar_urls(texto):
    padrao = r'https?://[^\s<>"]+|www\.[^\s<>"]+'
    return re.findall(padrao, texto)

texto = '''Este texto tem vários links, como https://www.google.com.br, www.google.com.br, https://www.youtube.com/watch?v=dQw4w9WgXcQ, http://www.youtube.com/watch?v=dQw4w9WgXcQ e https://youtu.be/dQw4w9WgXcQ'''
print(achar_urls(texto))