import re

def validar_email(email):
    padrão = r"^\w+\.?\w+@\w+\.?\w+\.\w+$" 
    # começar com letras, depois um ponto opcional, mais letras, depois @, mais letras,
    # depois um ponto opcional, mais letras, outro ponto opcional e termina com mais letras.
    # so o basico
    return re.match(pattern=padrão, string=email) is not None

email1 = "mario.patricio@ufc.br"
email2 = "mario.patricio@gmail.com"
email3 = "mario.patricio@gmail.com.br"
email4 = "mariopatricio@ufc.com"

# email invalido
email5 = "mario.patricio@ufc"

print(f"email1 {email1} valido:", validar_email(email1))
print(f"email2 {email2} valido:", validar_email(email2))
print(f"email3 {email3} valido:", validar_email(email3))
print(f"email4 {email4} valido:", validar_email(email4))
print(f"email5 {email5} valido:", validar_email(email5))


