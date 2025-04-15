import random
import json

print("hi")

def getal_in_woorden(getal):
    eenheden = ["", "een", "twee", "drie", "vier", "vijf", "zes", "zeven", "acht", "negen"]
    tientallen = ["", "", "twintig", "dertig", "veertig", "vijftig", "zestig", "zeventig", "tachtig", "negentig"]

    miljoentallen = ""
    duizendtallen = ""
    if getal >= 1000000:
        miljoentallen = getal_naar_honderden(getal // 1000000) + "miljoen "
        getal = getal % 1000000

    if getal >= 1000:
        duizendtallen = getal_naar_honderden(getal // 1000) + "duizend "
        getal = getal % 1000

    return miljoentallen + duizendtallen + getal_naar_honderden(getal)

def getal_naar_honderden(num):
    eenheden = ["", "een", "twee", "drie", "vier", "vijf", "zes", "zeven", "acht", "negen", "tien"]
    tientallen = ["", "", "twintig", "dertig", "veertig", "vijftig", "zestig", "zeventig", "tachtig", "negentig"]
    res = ""

    if num >= 100:
        res = eenheden[num // 100] + "honderd"
        num = num % 100

    if 10 < num < 20:
        special_cases = {
            11: "elf", 12: "twaalf", 13: "dertien", 14: "veertien", 15: "vijftien", 
            16: "zestien", 17: "zeventien", 18: "achttien", 19: "negentien"
        }
        res += special_cases[num]
    elif num >= 20:
        if num % 10 != 0:
            res += eenheden[num % 10] + "en"
        res += tientallen[num // 10]
    # elif num == 10:
    #     res += eenheden[0]
    else:
        print('num', num)
        res += eenheden[num]

    return res



if __name__ == "__main__":
 
    for d in range(10):

        case = {}

        getal = random.randint(1, 1000)

        print('getal', getal)

        woorden = getal_in_woorden(getal)
        print(woorden)


