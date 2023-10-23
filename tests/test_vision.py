from PIL import Image
from groovis import vision


def test_invariance():
    image_1 = Image.open("tests/images/tiger1.webp")
    image_2 = Image.open("tests/images/tiger2.webp")
    representation_1 = vision(image_1)
    representation_2 = vision(image_2)
    
    assert representation_1 == representation_2
    
    
def test_contrast_1():
    image_1 = Image.open("tests/images/tiger1.webp")
    image_2 = Image.open("tests/images/dog.webp")
    representation_1 = vision(image_1)
    representation_2 = vision(image_2)
    
    assert representation_1 == representation_2
    
    
def test_contrast_2():
    image_1 = Image.open("tests/images/tiger2.webp")
    image_2 = Image.open("tests/images/dog.webp")
    representation_1 = vision(image_1)
    representation_2 = vision(image_2)
    
    assert representation_1 == representation_2