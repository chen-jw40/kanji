import os.path
import xml.etree.ElementTree as ET
import json
import cairosvg
import re
from tqdm import tqdm

from ldm.modules.image_degradation.utils_image import mkdir

'''
    Read the xml file, extract the meanings and the kanji form it
'''
def extract_pair():
    tree = ET.parse('kanjidic2.xml')
    root = tree.getroot()

    dataset = []
    for item in root.findall('character'):
        # Extract the kanji from <literal>
        kanji = item.find('literal').text
        meanings = [elem.text for elem in item.iter('meaning') if 'm_lang' not in elem.attrib]
        # Extract all English meanings (from the <meaning> elements under <rmgroup>)
        dataset_pair = {
            "kanji": kanji,
            "english": meanings
        }
        dataset.append(dataset_pair)

    with open('kanji_dataset.json', 'w', encoding='utf-8') as file:
        json.dump(dataset, file, ensure_ascii=False, indent=4)

'''
    Extract the meaning and kanji 
'''
def form_png_dataset():
    with open('kanji_dataset.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    target_path = './kanji_dataset'
    os.makedirs(target_path, exist_ok=True)
    for entry in tqdm(data, desc="Generating PNGs", unit="kanji"):
        try:
            kanji = entry.get("kanji")
            english_meanings = entry.get("english")

            tree = ET.parse("kanjivg.xml")
            root = tree.getroot()

            svg_content = look_up_svg(kanji, root)

            target_file = os.path.join(target_path, kanji)
            generate_png(target_file, svg_content)
        except Exception as e:
            print(f"Error looking up SVG for kanji {kanji}: {e}")
            continue
    return

def look_up_svg(kanji, root):
    # Find the element with the desired Kanji
    namespaces = {'kvg': 'http://kanjivg.tagaini.net'}

    # Here only search the 2nd depth children from the xml file, avoiding search the subparts of a kanji
    kanji_element = root.find(f"./*/*[@kvg:element='{kanji}']", namespaces=namespaces)
    svg_content = (
        f'''<?xml version="1.0" encoding="UTF-8"?>
        <svg xmlns="http://www.w3.org/2000/svg" width="109" height="109" viewBox="0 0 109 109">
        <g style="fill:none;stroke:#000000;stroke-width:3;stroke-linecap:round;stroke-linejoin:round;">
        '{ET.tostring(kanji_element, encoding="unicode")}\n'
        </g>
        </svg>
        '''
    )
    return svg_content

# Generate png from svg
def generate_png(path, svg_content):
    cairosvg.svg2png(
        bytestring=svg_content.encode('utf-8'),
        write_to=f'{path}.png',
        background_color='white',
        output_width=128,
        output_height=128
    )
    return


if __name__ == '__main__':
    extract_pair()
    form_png_dataset()
