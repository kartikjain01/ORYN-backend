import re

class HinglishConverter:
    def __init__(self):
        # 🔥 Core phonetic mapping (keep small but powerful)
        self.word_map = {
            "kya": "क्या",
            "aap": "आप",
            "aapne": "आपने",
            "kabhi": "कभी",
            "socha": "सोचा",
            "hai": "है",
            "hain": "हैं",
            "tha": "था",
            "thi": "थी",
            "the": "थे",
            "ki": "कि",
            "ka": "का",
            "ke": "के",
            "ko": "को",
            "se": "से",
            "mein": "में",
            "jab": "जब",
            "aur": "और",
            "ek": "एक",
            "aisi": "ऐसी",
            "duniya": "दुनिया",
            "jahan": "जहाँ",
            "sirf": "सिर्फ",
            "nahi": "नहीं",
            "pesh": "पेश",
            "hum": "हम",
            "karte": "करते",
            "karta": "करता",
            "karna": "करना",
            "unhe": "उन्हें",
            "dete": "देते",
            "chahe": "चाहे",
            "woh": "वह",
            "ya": "या",
            "phir": "फिर",
            "nayi": "नई",
            "ab": "अब",
            "aapko": "आपको",
            "zaroorat": "ज़रूरत",
            "bas": "बस",
            "apni": "अपनी",
            "apna": "अपना",
            "apne": "अपने",

            # 🔥 IMPORTANT missing words (from your debug)
            "toh": "तो",
            "to": "तो",
            "hota": "होता",
            "hoti": "होती",
            "hote": "होते",
            "saath": "साथ",
            "milte": "मिलते",
            "milti": "मिलती",
            "milta": "मिलता",
            "hamari": "हमारी",
            "aapke": "आपके",
            "karein": "करें",
            "karo": "करो",
            "goonjte": "गूंजते",
            "huye": "हुए",
            "dekhein": "देखें",
            "rahe": "रहे",
            "rahi": "रही",
            "raha": "रहा",
        }

    def is_hinglish(self, text):
        # 🔥 improved detection (allow mixed but prioritize Hinglish)
        has_english = re.search(r'[a-zA-Z]', text)
        return bool(has_english)

    def convert(self, text):
        words = text.split()
        converted = []

        for word in words:
            # separate punctuation
            prefix = re.match(r'^\W+', word)
            suffix = re.search(r'\W+$', word)

            core = re.sub(r'^\W+|\W+$', '', word.lower())

            if core in self.word_map:
                new_word = self.word_map[core]
            else:
                new_word = word  # keep original if unknown

            # reattach punctuation safely
            if prefix:
                new_word = prefix.group() + new_word
            if suffix:
                new_word = new_word + suffix.group()

            converted.append(new_word)

        return " ".join(converted)
