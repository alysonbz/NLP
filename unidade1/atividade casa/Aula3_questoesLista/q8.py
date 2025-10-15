# 8. Validação de datas: dd/mm/aaaa

import re

datas = ["31/02/2023", "29/02/2024", "12/13/2020", "01/01/1999"]
comp_data = re.compile(r"^(0[1-9]|[12]\d|3[01])/(0[1-9]|1[0-2])/\d{4}$")

for i in datas:
    print(f"As datas válidas são {comp_data.match(i)}")
