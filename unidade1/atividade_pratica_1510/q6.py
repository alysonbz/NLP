import re 

def vefificar_senha_segura(txt):
    if len(txt) < 8:
        return False

    if re.search(pattern=r"[A-Z]", string=txt) is None:
        # print("entrou")
        return False
    
    if re.search(pattern=r"[a-z]", string=txt) is None:
        return False
    
    if re.search(pattern=r"[0-9]", string=txt) is None:
        return False
    
    if re.search(pattern=r"[!@#$%^&*()]", string=txt) is None:
        return False
    
    return True

print(vefificar_senha_segura("12345678"))# so numero
print(vefificar_senha_segura("tEste1234")) # sem caracteres especiais
print(vefificar_senha_segura("!teste12355"))# sem maiusculas
print(vefificar_senha_segura("TESTE!@12355")) # sem minusculas
print(vefificar_senha_segura("tEste@1234")) # correta