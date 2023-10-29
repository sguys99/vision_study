from PIL import Image
from groovis import vision


def test_invariance():
    image_tiger_1 = Image.open("tests/images/tiger1.webp")
    image_tiger_2 = Image.open("tests/images/tiger2.webp")
    image_dog = Image.open("tests/images/dog.webp")
    
    tiger_1 = vision(image_tiger_1)
    tiger_2 = vision(image_tiger_2)
    dog = vision(image_dog)
    
    diff_tiger_tiger = tiger_2 - tiger_2
    diff_tiger_dog_1 = tiger_1 - dog
    diff_tiger_dog_2 = tiger_2 - dog
    
    assert (diff_tiger_dog_1 - diff_tiger_dog_2) / 2 > diff_tiger_tiger
    