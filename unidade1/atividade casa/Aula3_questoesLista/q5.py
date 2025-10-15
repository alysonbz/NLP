# 5. Extração de URLs

import re

url = "Visite https://example.com e também www.site.org/path?x=1 ou http://foo.bar."

url_val = re.findall(r"(?:https?:// | www\.)[^\s<>()\[\]]+", url)

print(f"O texto é: {url}")
print(f"\nMas as urls válidas para o teste são: {url_val}")