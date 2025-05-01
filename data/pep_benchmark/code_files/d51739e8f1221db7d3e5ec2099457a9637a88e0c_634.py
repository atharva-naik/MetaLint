from flask import Flask
import random

app = Flask(__name__)
#Rótin
@app.route("/")
def root():
    return ("<a href='Subpage1'>Undirsíða 1</a> <a href='Subpage2'>Undirsíða 2</a>")
        
@app.route("/Subpage1/")
def about():
    brandarar = ["1", "2", "3", "4", "5"]
    brandari = ""

    randomtala = (random.randint(1, 5))
    if randomtala == 1:
        brandari = "Veistu hvað sagði sólpallurinn sagði við hinn sólpallinn? Ég vildi að við værum svlair!"
    elif randomtala == 2:
        brandari = "Einu sinni var pabbi að þvo bílinn með syni sínum svo stoppa þeir í smá og sonurinn spyr “pabbi, getum við notað svampa?”"
    elif randomtala == 3:
        brandari = "Gúrkur eru góðar fyrir minnið. Vinur minn fékk eina upp í rassin og man vel eftir því."
    elif randomtala == 4:
        brandari = "Hefur þú heyrt um minka búið sem minnkaði og minnkaði þar til það var búið?"
    elif randomtala == 5:
        brandari = "Afhverju sérðu aldrei fíla fela sig í trjám? Því þeir eru að fela sig..."
    return (brandari)

@app.route('/Subpage2/<name_id>')
def blog(name_id):
    return "This is blog post number " + str(name_id)


if __name__ == "__main__":
    app.run(debug=True)