"""Support for ELEVATER benchmark datasets including CIFAR-10/100."""

from typing import Dict, List, Optional, Any
import json
import os
import random
import numpy as np
import urllib.error
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import DataLoader

from ..core.base import BaseDataset
from ..core.exceptions import DatasetError
from ..retrieval.cache_manager import sanitize_filename


class ELEVATERDataset(BaseDataset):
    """
    Handler for ELEVATER benchmark datasets.
    
    ELEVATER (Evaluation of Language-augmented Visual Task Adaptation with Eye-Tracking)
    includes multiple vision datasets for comprehensive evaluation.
    """
    
    # Mapping of ELEVATER dataset names to their properties
    ELEVATER_DATASETS = {
        'caltech101': {
            'num_classes': 101,
            'task': 'classification',
            'type': 'file_based',
            'class_names': [
                'accordion',
                'airplanes',
                'anchor',
                'ant',
                'barrel',
                'bass',
                'beaver',
                'binocular',
                'bonsai',
                'brain',
                'brontosaurus',
                'buddha',
                'butterfly',
                'camera',
                'cannon',
                'car_side',
                'ceiling_fan',
                'cellphone',
                'chair',
                'chandelier',
                'cougar_body',
                'cougar_face',
                'crab',
                'crayfish',
                'crocodile',
                'crocodile_head',
                'cup',
                'dalmatian',
                'dollar_bill',
                'dolphin',
                'dragonfly',
                'electric_guitar',
                'elephant',
                'emu',
                'euphonium',
                'ewer',
                'faces_easy',
                'faces',
                'ferry',
                'flamingo',
                'flamingo_head',
                'garfield',
                'gerenuk',
                'gramophone',
                'grand_piano',
                'hawksbill',
                'headphone',
                'hedgehog',
                'helicopter',
                'ibis',
                'inline_skate',
                'joshua_tree',
                'kangaroo',
                'ketch',
                'lamp',
                'laptop',
                'leopards',
                'llama',
                'lobster',
                'lotus',
                'mandolin',
                'mayfly',
                'menorah',
                'metronome',
                'minaret',
                'motorbikes',
                'nautilus',
                'octopus',
                'okapi',
                'pagoda',
                'panda',
                'pigeon',
                'pizza',
                'platypus',
                'pyramid',
                'revolver',
                'rhino',
                'rooster',
                'saxophone',
                'schooner',
                'scissors',
                'scorpion',
                'sea_horse',
                'snoopy',
                'soccer_ball',
                'stapler',
                'starfish',
                'stegosaurus',
                'stop_sign',
                'strawberry',
                'sunflower',
                'tick',
                'trilobite',
                'umbrella',
                'watch',
                'water_lilly',
                'wheelchair',
                'wild_cat',
                'windsor_chair',
                'wrench',
                'yin_yang'
            ]
        },
        'cifar10': {
            'num_classes': 10,
            'task': 'classification',
            'type': 'torchvision',
            'class_names': [
                'airplane',
                'automobile',
                'bird',
                'cat',
                'deer',
                'dog',
                'frog',
                'horse',
                'ship',
                'truck'
            ]
        },
        'cifar100': {
            'num_classes': 100,
            'task': 'classification',
            'type': 'torchvision',
            # Class names in torchvision's CIFAR-100 ordering (alphabetically sorted)
            # Converted from underscores to spaces to match REACT style
            'class_names': [
                'apple', 'aquarium fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
                'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
                'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
                'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
                'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
                'house', 'kangaroo', 'keyboard', 'lamp', 'lawn mower', 'leopard', 'lion',
                'lizard', 'lobster', 'man', 'maple tree', 'motorcycle', 'mountain', 'mouse',
                'mushroom', 'oak tree', 'orange', 'orchid', 'otter', 'palm tree', 'pear',
                'pickup truck', 'pine tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum',
                'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark',
                'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel',
                'streetcar', 'sunflower', 'sweet pepper', 'table', 'tank', 'telephone',
                'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle',
                'wardrobe', 'whale', 'willow tree', 'wolf', 'woman', 'worm'
            ]
        },
        'country211': {
            'num_classes': 211,
            'task': 'classification',
            'type': 'file_based',
            'class_names': [
                'Andorra',
                'United Arab Emirates',
                'Afghanistan',
                'Antigua and Barbuda',
                'Anguilla',
                'Albania',
                'Armenia',
                'Angola',
                'Antarctica',
                'Argentina',
                'Austria',
                'Australia',
                'Aruba',
                'Aland Islands',
                'Azerbaijan',
                'Bosnia and Herzegovina',
                'Barbados',
                'Bangladesh',
                'Belgium',
                'Burkina Faso',
                'Bulgaria',
                'Bahrain',
                'Benin',
                'Bermuda',
                'Brunei Darussalam',
                'Bolivia',
                'Bonaire, Saint Eustatius and Saba',
                'Brazil',
                'Bahamas',
                'Bhutan',
                'Botswana',
                'Belarus',
                'Belize',
                'Canada',
                'DR Congo',
                'Central African Republic',
                'Switzerland',
                "Cote d'Ivoire",
                'Cook Islands',
                'Chile',
                'Cameroon',
                'China',
                'Colombia',
                'Costa Rica',
                'Cuba',
                'Cabo Verde',
                'Curacao',
                'Cyprus',
                'Czech Republic',
                'Germany',
                'Denmark',
                'Dominica',
                'Dominican Republic',
                'Algeria',
                'Ecuador',
                'Estonia',
                'Egypt',
                'Spain',
                'Ethiopia',
                'Finland',
                'Fiji',
                'Falkland Islands',
                'Faeroe Islands',
                'France',
                'Gabon',
                'United Kingdom',
                'Grenada',
                'Georgia',
                'French Guiana',
                'Guernsey',
                'Ghana',
                'Gibraltar',
                'Greenland',
                'Gambia',
                'Guadeloupe',
                'Greece',
                'South Georgia and South Sandwich Is.',
                'Guatemala',
                'Guam',
                'Guyana',
                'Hong Kong',
                'Honduras',
                'Croatia',
                'Haiti',
                'Hungary',
                'Indonesia',
                'Ireland',
                'Israel',
                'Isle of Man',
                'India',
                'Iraq',
                'Iran',
                'Iceland',
                'Italy',
                'Jersey',
                'Jamaica',
                'Jordan',
                'Japan',
                'Kenya',
                'Kyrgyz Republic',
                'Cambodia',
                'St. Kitts and Nevis',
                'North Korea',
                'South Korea',
                'Kuwait',
                'Cayman Islands',
                'Kazakhstan',
                'Laos',
                'Lebanon',
                'St. Lucia',
                'Liechtenstein',
                'Sri Lanka',
                'Liberia',
                'Lithuania',
                'Luxembourg',
                'Latvia',
                'Libya',
                'Morocco',
                'Monaco',
                'Moldova',
                'Montenegro',
                'Saint-Martin',
                'Madagascar',
                'Macedonia',
                'Mali',
                'Myanmar',
                'Mongolia',
                'Macau',
                'Martinique',
                'Mauritania',
                'Malta',
                'Mauritius',
                'Maldives',
                'Malawi',
                'Mexico',
                'Malaysia',
                'Mozambique',
                'Namibia',
                'New Caledonia',
                'Nigeria',
                'Nicaragua',
                'Netherlands',
                'Norway',
                'Nepal',
                'New Zealand',
                'Oman',
                'Panama',
                'Peru',
                'French Polynesia',
                'Papua New Guinea',
                'Philippines',
                'Pakistan',
                'Poland',
                'Puerto Rico',
                'Palestine',
                'Portugal',
                'Palau',
                'Paraguay',
                'Qatar',
                'Reunion',
                'Romania',
                'Serbia',
                'Russia',
                'Rwanda',
                'Saudi Arabia',
                'Solomon Islands',
                'Seychelles',
                'Sudan',
                'Sweden',
                'Singapore',
                'St. Helena',
                'Slovenia',
                'Svalbard and Jan Mayen Islands',
                'Slovakia',
                'Sierra Leone',
                'San Marino',
                'Senegal',
                'Somalia',
                'South Sudan',
                'El Salvador',
                'Sint Maarten',
                'Syria',
                'Eswatini',
                'Togo',
                'Thailand',
                'Tajikistan',
                'Timor-Leste',
                'Turkmenistan',
                'Tunisia',
                'Tonga',
                'Turkey',
                'Trinidad and Tobago',
                'Taiwan',
                'Tanzania',
                'Ukraine',
                'Uganda',
                'United States',
                'Uruguay',
                'Uzbekistan',
                'Vatican',
                'Venezuela',
                'British Virgin Islands',
                'United States Virgin Islands',
                'Vietnam',
                'Vanuatu',
                'Samoa',
                'Kosovo',
                'Yemen',
                'South Africa',
                'Zambia',
                'Zimbabwe'
            ]
        },
        'dtd': {
            'num_classes': 47,
            'task': 'classification',
            'type': 'torchvision',
            'class_names': [
                'banded',
                'blotchy',
                'braided',
                'bubbly',
                'bumpy',
                'chequered',
                'cobwebbed',
                'cracked',
                'crosshatched',
                'crystalline',
                'dotted',
                'fibrous',
                'flecked',
                'freckled',
                'frilly',
                'gauzy',
                'grid',
                'grooved',
                'honeycombed',
                'interlaced',
                'knitted',
                'lacelike',
                'lined',
                'marbled',
                'matted',
                'meshed',
                'paisley',
                'perforated',
                'pitted',
                'pleated',
                'polka-dotted',
                'porous',
                'potholed',
                'scaly',
                'smeared',
                'spiralled',
                'sprinkled',
                'stained',
                'stratified',
                'striped',
                'studded',
                'swirly',
                'veined',
                'waffled',
                'woven',
                'wrinkled',
                'zigzagged'
            ]
        },
        'eurosat': {
            'num_classes': 10,
            'task': 'classification',
            'type': 'torchvision',
            'class_names': [
                'annual crop land',
                'forest',
                'brushland or shrubland',
                'highway or road',
                'industrial buildings or commercial buildings',
                'pasture land',
                'permanent crop land',
                'residential buildings or homes or apartments',
                'river',
                'lake or sea'
            ]
        },
        'fer2013': {
            'num_classes': 7,
            'task': 'classification',
            'type': 'file_based',
            'class_names': [
                ['angry'],
                ['disgusted'],
                ['fearful'],
                ['happy', 'smiling'],
                ['neutral', 'bored'],
                ['sad', 'depressed'],
                ['surprised', 'shocked', 'spooked']
            ]
        },
        'fgvc_aircraft': {
            'num_classes': 100,
            'task': 'classification',
            'type': 'torchvision',
            'class_names':[
                '707-320',
                '727-200',
                '737-200',
                '737-300',
                '737-400',
                '737-500',
                '737-600',
                '737-700',
                '737-800',
                '737-900',
                '747-100',
                '747-200',
                '747-300',
                '747-400',
                '757-200',
                '757-300',
                '767-200',
                '767-300',
                '767-400',
                '777-200',
                '777-300',
                'A300B4',
                'A310',
                'A318',
                'A319',
                'A320',
                'A321',
                'A330-200',
                'A330-300',
                'A340-200',
                'A340-300',
                'A340-500',
                'A340-600',
                'A380',
                'An-12',
                'ATR-42',
                'ATR-72',
                'BAE 146-200',
                'BAE 146-300',
                'BAE-125',
                'Beechcraft 1900',
                'Boeing 717',
                'C-130',
                'C-47',
                'Cessna 172',
                'Cessna 208',
                'Cessna 525',
                'Cessna 560',
                'Challenger 600',
                'CRJ-200',
                'CRJ-700',
                'CRJ-900',
                'DC-10',
                'DC-3',
                'DC-6',
                'DC-8',
                'DC-9-30',
                'DH-82',
                'DHC-1',
                'DHC-6',
                'DHC-8-100',
                'DHC-8-300',
                'Dornier 328',
                'DR-400',
                'E-170',
                'E-190',
                'E-195',
                'EMB-120',
                'Embraer Legacy 600',
                'ERJ 135',
                'ERJ 145',
                'Eurofighter Typhoon',
                'F-16A-B',
                'F-A-18',
                'Falcon 2000',
                'Falcon 900',
                'Fokker 100',
                'Fokker 50',
                'Fokker 70',
                'Global Express',
                'Gulfstream IV',
                'Gulfstream V',
                'Hawk T1',
                'Il-76',
                'L-1011',
                'MD-11',
                'MD-80',
                'MD-87',
                'MD-90',
                'Metroliner',
                'Model B200',
                'PA-28',
                'Saab 2000',
                'Saab 340',
                'Spitfire',
                'SR-20',
                'Tornado',
                'Tu-134',
                'Tu-154',
                'Yak-42'
            ]
        },
        'food101': {
            'num_classes': 101,
            'task': 'classification',
            'type': 'torchvision',
            'class_names': [
                'apple pie',
                'baby back ribs',
                'baklava',
                'beef carpaccio',
                'beef tartare',
                'beet salad',
                'beignets',
                'bibimbap',
                'bread pudding',
                'breakfast burrito',
                'bruschetta',
                'caesar salad',
                'cannoli',
                'caprese salad',
                'carrot cake',
                'ceviche',
                'cheesecake',
                'cheese plate',
                'chicken curry',
                'chicken quesadilla',
                'chicken wings',
                'chocolate cake',
                'chocolate mousse',
                'churros',
                'clam chowder',
                'club sandwich',
                'crab cakes',
                'creme brulee',
                'croque madame',
                'cup cakes',
                'deviled eggs',
                'donuts',
                'dumplings',
                'edamame',
                'eggs benedict',
                'escargots',
                'falafel',
                'filet mignon',
                'fish and chips',
                'foie gras',
                'french fries',
                'french onion soup',
                'french toast',
                'fried calamari',
                'fried rice',
                'frozen yogurt',
                'garlic bread',
                'gnocchi',
                'greek salad',
                'grilled cheese sandwich',
                'grilled salmon',
                'guacamole',
                'gyoza',
                'hamburger',
                'hot and sour soup',
                'hot dog',
                'huevos rancheros',
                'hummus',
                'ice cream',
                'lasagna',
                'lobster bisque',
                'lobster roll sandwich',
                'macaroni and cheese',
                'macarons',
                'miso soup',
                'mussels',
                'nachos',
                'omelette',
                'onion rings',
                'oysters',
                'pad thai',
                'paella',
                'pancakes',
                'panna cotta',
                'peking duck',
                'pho',
                'pizza',
                'pork chop',
                'poutine',
                'prime rib',
                'pulled pork sandwich',
                'ramen',
                'ravioli',
                'red velvet cake',
                'risotto',
                'samosa',
                'sashimi',
                'scallops',
                'seaweed salad',
                'shrimp and grits',
                'spaghetti bolognese',
                'spaghetti carbonara',
                'spring rolls',
                'steak',
                'strawberry shortcake',
                'sushi',
                'tacos',
                'takoyaki',
                'tiramisu',
                'tuna tartare',
                'waffles'
            ]
        },
        'gtsrb': {
            'num_classes': 43,
            'task': 'classification',
            'type': 'torchvision',
            'class_names': [
                'red and white circle 20 kph speed limit',
                'red and white circle 30 kph speed limit',
                'red and white circle 50 kph speed limit',
                'red and white circle 60 kph speed limit',
                'red and white circle 70 kph speed limit',
                'red and white circle 80 kph speed limit',
                'end / de-restriction of 80 kph speed limit',
                'red and white circle 100 kph speed limit',
                'red and white circle 120 kph speed limit',
                'red and white circle red car and black car no passing',
                'red and white circle red truck and black car no passing',
                'red and white triangle road intersection warning',
                'white and yellow diamond priority road',
                'red and white upside down triangle yield right-of-way',
                'stop',
                'empty red and white circle',
                'red and white circle no truck entry',
                'red circle with white horizonal stripe no entry',
                'red and white triangle with exclamation mark warning',
                'red and white triangle with black left curve approaching warning',
                'red and white triangle with black right curve approaching warning',
                'red and white triangle with black double curve approaching warning',
                'red and white triangle rough / bumpy road warning',
                'red and white triangle car skidding / slipping warning',
                'red and white triangle with merging / narrow lanes warning',
                'red and white triangle with person digging / construction / road work warning',
                'red and white triangle with traffic light approaching warning',
                'red and white triangle with person walking warning',
                'red and white triangle with child and person walking warning',
                'red and white triangle with bicyle warning',
                'red and white triangle with snowflake / ice warning',
                'red and white triangle with deer warning',
                'white circle with gray strike bar no speed limit',
                'blue circle with white right turn arrow mandatory',
                'blue circle with white left turn arrow mandatory',
                'blue circle with white forward arrow mandatory',
                'blue circle with white forward or right turn arrow mandatory',
                'blue circle with white forward or left turn arrow mandatory',
                'blue circle with white keep right arrow mandatory',
                'blue circle with white keep left arrow mandatory',
                'blue circle with white arrows indicating a traffic circle',
                'white circle with gray strike bar indicating no passing for cars has ended',
                'white circle with gray strike bar indicating no passing for trucks has ended'
            ]
        },
        'hateful_memes': {
            'num_classes': 2,
            'task': 'classification',
            'type': 'file_based',
            'class_names': [
                'meme',
                'hatespeech meme',
            ]
        },
        'kitti_distance': {
            'num_classes': 4,
            'task': 'classification',
            'type': 'file_based',
            'class_names': [
                'a photo i took of a car on my left or right side.',
                'a photo i took with a car nearby.',
                'a photo i took with a car in the distance.',
                'a photo i took with no car.'
            ]
        },
        'mnist': {
            'num_classes': 10,
            'task': 'classification',
            'type': 'torchvision',
            'class_names': [
                '0',
                '1',
                '2',
                '3',
                '4',
                '5',
                '6',
                '7',
                '8',
                '9'
            ]
        },
        'oxford_flowers102': {
            'num_classes': 102,
            'task': 'classification',
            'type': 'torchvision',
            'class_names': [
                'pink primrose',
                'hard-leaved pocket orchid',
                'canterbury bells',
                'sweet pea',
                'english marigold',
                'tiger lily',
                'moon orchid',
                'bird of paradise',
                'monkshood',
                'globe thistle',
                'snapdragon',
                "colt's foot",
                'king protea',
                'spear thistle',
                'yellow iris',
                'globe flower',
                'purple coneflower',
                'peruvian lily',
                'balloon flower',
                'giant white arum lily',
                'fire lily',
                'pincushion flower',
                'fritillary',
                'red ginger',
                'grape hyacinth',
                'corn poppy',
                'prince of wales feathers',
                'stemless gentian',
                'artichoke',
                'sweet william',
                'carnation',
                'garden phlox',
                'love in the mist',
                'mexican aster',
                'alpine sea holly',
                'ruby-lipped cattleya',
                'cape flower',
                'great masterwort',
                'siam tulip',
                'lenten rose',
                'barbeton daisy',
                'daffodil',
                'sword lily',
                'poinsettia',
                'bolero deep blue',
                'wallflower',
                'marigold',
                'buttercup',
                'oxeye daisy',
                'common dandelion',
                'petunia',
                'wild pansy',
                'primula',
                'sunflower',
                'pelargonium',
                'bishop of llandaff',
                'gaura',
                'geranium',
                'orange dahlia',
                'pink and yellow dahlia',
                'cautleya spicata',
                'japanese anemone',
                'black-eyed susan',
                'silverbush',
                'californian poppy',
                'osteospermum',
                'spring crocus',
                'bearded iris',
                'windflower',
                'tree poppy',
                'gazania',
                'azalea',
                'water lily',
                'rose',
                'thorn apple',
                'morning glory',
                'passion flower',
                'lotus',
                'toad lily',
                'anthurium',
                'frangipani',
                'clematis',
                'hibiscus',
                'columbine',
                'desert-rose',
                'tree mallow',
                'magnolia',
                'cyclamen',
                'watercress',
                'canna lily',
                'hippeastrum',
                'bee balm',
                'air plant',
                'foxglove',
                'bougainvillea',
                'camellia',
                'mallow',
                'mexican petunia',
                'bromelia',
                'blanket flower',
                'trumpet creeper',
                'blackberry lily'
            ]
        },
        'oxford_pets': {
            'num_classes': 37,
            'task': 'classification',
            'type': 'torchvision',
            'class_names': [
                'Abyssinian',
                'american bulldog',
                'american pit bull terrier',
                'basset hound',
                'beagle',
                'Bengal',
                'Birman',
                'Bombay',
                'boxer',
                'British Shorthair',
                'chihuahua',
                'Egyptian Mau',
                'english cocker spaniel',
                'english setter',
                'german shorthaired',
                'great pyrenees',
                'havanese',
                'japanese chin',
                'keeshond',
                'leonberger',
                'Maine Coon',
                'miniature pinscher',
                'newfoundland',
                'Persian',
                'pomeranian',
                'pug',
                'Ragdoll',
                'Russian Blue',
                'saint bernard',
                'samoyed',
                'scottish terrier',
                'shiba inu',
                'Siamese',
                'Sphynx',
                'staffordshire bull terrier',
                'wheaten terrier',
                'yorkshire terrier'
            ]
        },
        'patchcamelyon': {
            'num_classes': 2,
            'task': 'classification',
            'type': 'file_based',
            'class_names':[
                'lymph node',
                'lymph node containing metastatic tumor tissue',
            ]
        },
        'rendered_sst2': {
            'num_classes': 2,
            'task': 'classification',
            'type': 'file_based',
            'class_names': [
                'negative',
                'positive'
            ]
        },
        'resisc45': {
            'num_classes': 45,
            'task': 'classification',
            'type': 'file_based',
            'class_names': [
                'airplane',
                'airport',
                'baseball diamond',
                'basketball court',
                'beach',
                'bridge',
                'chaparral',
                'church',
                'circular farmland',
                'cloud',
                'commercial area',
                'dense residential',
                'desert',
                'forest',
                'freeway',
                'golf course',
                'ground track field',
                'harbor',
                'industrial area',
                'intersection',
                'island',
                'lake',
                'meadow',
                'medium residential',
                'mobile home park',
                'mountain',
                'overpass',
                'palace',
                'parking lot',
                'railway',
                'railway station',
                'rectangular farmland',
                'river',
                'roundabout',
                'runway',
                'sea ice',
                'ship',
                'snowberg',
                'sparse residential',
                'stadium',
                'storage tank',
                'tennis court',
                'terrace',
                'thermal power station',
                'wetland'
            ]
        },
        'stanford_cars': {
            'num_classes': 196,
            'task': 'classification',
            'type': 'file_based',
            'class_names': [
                'Acura Integra Type R 2001',
                'Acura RL Sedan 2012',
                'Acura TL Sedan 2012',
                'Acura TL Type-S 2008',
                'Acura TSX Sedan 2012',
                'Acura ZDX Hatchback 2012',
                'AM General Hummer SUV 2000',
                'Aston Martin V8 Vantage Convertible 2012',
                'Aston Martin V8 Vantage Coupe 2012',
                'Aston Martin Virage Convertible 2012',
                'Aston Martin Virage Coupe 2012',
                'Audi 100 Sedan 1994',
                'Audi 100 Wagon 1994',
                'Audi A5 Coupe 2012',
                'Audi R8 Coupe 2012',
                'Audi RS 4 Convertible 2008',
                'Audi S4 Sedan 2007',
                'Audi S4 Sedan 2012',
                'Audi S5 Convertible 2012',
                'Audi S5 Coupe 2012',
                'Audi S6 Sedan 2011',
                'Audi TT Hatchback 2011',
                'Audi TT RS Coupe 2012',
                'Audi TTS Coupe 2012',
                'Audi V8 Sedan 1994',
                'Bentley Arnage Sedan 2009',
                'Bentley Continental Flying Spur Sedan 2007',
                'Bentley Continental GT Coupe 2007',
                'Bentley Continental GT Coupe 2012',
                'Bentley Continental Supersports Conv. Convertible 2012',
                'Bentley Mulsanne Sedan 2011',
                'BMW 1 Series Convertible 2012',
                'BMW 1 Series Coupe 2012',
                'BMW 3 Series Sedan 2012',
                'BMW 3 Series Wagon 2012',
                'BMW 6 Series Convertible 2007',
                'BMW ActiveHybrid 5 Sedan 2012',
                'BMW M3 Coupe 2012',
                'BMW M5 Sedan 2010',
                'BMW M6 Convertible 2010',
                'BMW X3 SUV 2012',
                'BMW X5 SUV 2007',
                'BMW X6 SUV 2012',
                'BMW Z4 Convertible 2012',
                'Bugatti Veyron 16.4 Convertible 2009',
                'Bugatti Veyron 16.4 Coupe 2009',
                'Buick Enclave SUV 2012',
                'Buick Rainier SUV 2007',
                'Buick Regal GS 2012',
                'Buick Verano Sedan 2012',
                'Cadillac CTS-V Sedan 2012',
                'Cadillac Escalade EXT Crew Cab 2007',
                'Cadillac SRX SUV 2012',
                'Chevrolet Avalanche Crew Cab 2012',
                'Chevrolet Camaro Convertible 2012',
                'Chevrolet Cobalt SS 2010',
                'Chevrolet Corvette Convertible 2012',
                'Chevrolet Corvette Ron Fellows Edition Z06 2007',
                'Chevrolet Corvette ZR1 2012',
                'Chevrolet Express Cargo Van 2007',
                'Chevrolet Express Van 2007',
                'Chevrolet HHR SS 2010',
                'Chevrolet Impala Sedan 2007',
                'Chevrolet Malibu Hybrid Sedan 2010',
                'Chevrolet Malibu Sedan 2007',
                'Chevrolet Monte Carlo Coupe 2007',
                'Chevrolet Silverado 1500 Classic Extended Cab 2007',
                'Chevrolet Silverado 1500 Extended Cab 2012',
                'Chevrolet Silverado 1500 Hybrid Crew Cab 2012',
                'Chevrolet Silverado 1500 Regular Cab 2012',
                'Chevrolet Silverado 2500HD Regular Cab 2012',
                'Chevrolet Sonic Sedan 2012',
                'Chevrolet Tahoe Hybrid SUV 2012',
                'Chevrolet TrailBlazer SS 2009',
                'Chevrolet Traverse SUV 2012',
                'Chrysler 300 SRT-8 2010',
                'Chrysler Aspen SUV 2009',
                'Chrysler Crossfire Convertible 2008',
                'Chrysler PT Cruiser Convertible 2008',
                'Chrysler Sebring Convertible 2010',
                'Chrysler Town and Country Minivan 2012',
                'Daewoo Nubira Wagon 2002',
                'Dodge Caliber Wagon 2007',
                'Dodge Caliber Wagon 2012',
                'Dodge Caravan Minivan 1997',
                'Dodge Challenger SRT8 2011',
                'Dodge Charger Sedan 2012',
                'Dodge Charger SRT-8 2009',
                'Dodge Dakota Club Cab 2007',
                'Dodge Dakota Crew Cab 2010',
                'Dodge Durango SUV 2007',
                'Dodge Durango SUV 2012',
                'Dodge Journey SUV 2012',
                'Dodge Magnum Wagon 2008',
                'Dodge Ram Pickup 3500 Crew Cab 2010',
                'Dodge Ram Pickup 3500 Quad Cab 2009',
                'Dodge Sprinter Cargo Van 2009',
                'Eagle Talon Hatchback 1998',
                'Ferrari 458 Italia Convertible 2012',
                'Ferrari 458 Italia Coupe 2012',
                'Ferrari California Convertible 2012',
                'Ferrari FF Coupe 2012',
                'FIAT 500 Abarth 2012',
                'FIAT 500 Convertible 2012',
                'Fisker Karma Sedan 2012',
                'Ford E-Series Wagon Van 2012',
                'Ford Edge SUV 2012',
                'Ford Expedition EL SUV 2009',
                'Ford F-150 Regular Cab 2007',
                'Ford F-150 Regular Cab 2012',
                'Ford F-450 Super Duty Crew Cab 2012',
                'Ford Fiesta Sedan 2012',
                'Ford Focus Sedan 2007',
                'Ford Freestar Minivan 2007',
                'Ford GT Coupe 2006',
                'Ford Mustang Convertible 2007',
                'Ford Ranger SuperCab 2011',
                'Geo Metro Convertible 1993',
                'GMC Acadia SUV 2012',
                'GMC Canyon Extended Cab 2012',
                'GMC Savana Van 2012',
                'GMC Terrain SUV 2012',
                'GMC Yukon Hybrid SUV 2012',
                'Honda Accord Coupe 2012',
                'Honda Accord Sedan 2012',
                'Honda Odyssey Minivan 2007',
                'Honda Odyssey Minivan 2012',
                'HUMMER H2 SUT Crew Cab 2009',
                'HUMMER H3T Crew Cab 2010',
                'Hyundai Accent Sedan 2012',
                'Hyundai Azera Sedan 2012',
                'Hyundai Elantra Sedan 2007',
                'Hyundai Elantra Touring Hatchback 2012',
                'Hyundai Genesis Sedan 2012',
                'Hyundai Santa Fe SUV 2012',
                'Hyundai Sonata Hybrid Sedan 2012',
                'Hyundai Sonata Sedan 2012',
                'Hyundai Tucson SUV 2012',
                'Hyundai Veloster Hatchback 2012',
                'Hyundai Veracruz SUV 2012',
                'Infiniti G Coupe IPL 2012',
                'Infiniti QX56 SUV 2011',
                'Isuzu Ascender SUV 2008',
                'Jaguar XK XKR 2012',
                'Jeep Compass SUV 2012',
                'Jeep Grand Cherokee SUV 2012',
                'Jeep Liberty SUV 2012',
                'Jeep Patriot SUV 2012',
                'Jeep Wrangler SUV 2012',
                'Lamborghini Aventador Coupe 2012',
                'Lamborghini Diablo Coupe 2001',
                'Lamborghini Gallardo LP 570-4 Superleggera 2012',
                'Lamborghini Reventon Coupe 2008',
                'Land Rover LR2 SUV 2012',
                'Land Rover Range Rover SUV 2012',
                'Lincoln Town Car Sedan 2011',
                'Maybach Landaulet Convertible 2012',
                'Mazda Tribute SUV 2011',
                'McLaren MP4-12C Coupe 2012',
                'Mercedes-Benz 300-Class Convertible 1993',
                'Mercedes-Benz C-Class Sedan 2012',
                'Mercedes-Benz E-Class Sedan 2012',
                'Mercedes-Benz S-Class Sedan 2012',
                'Mercedes-Benz SL-Class Coupe 2009',
                'Mercedes-Benz Sprinter Van 2012',
                'MINI Cooper Roadster Convertible 2012',
                'Mitsubishi Lancer Sedan 2012',
                'Nissan 240SX Coupe 1998',
                'Nissan Juke Hatchback 2012',
                'Nissan Leaf Hatchback 2012',
                'Nissan NV Passenger Van 2012',
                'Plymouth Neon Coupe 1999',
                'Porsche Panamera Sedan 2012',
                'Ram C-V Cargo Van Minivan 2012',
                'Rolls-Royce Ghost Sedan 2012',
                'Rolls-Royce Phantom Drophead Coupe Convertible 2012',
                'Rolls-Royce Phantom Sedan 2012',
                'Scion xD Hatchback 2012',
                'smart fortwo Convertible 2012',
                'Spyker C8 Convertible 2009',
                'Spyker C8 Coupe 2009',
                'Suzuki Aerio Sedan 2007',
                'Suzuki Kizashi Sedan 2012',
                'Suzuki SX4 Hatchback 2012',
                'Suzuki SX4 Sedan 2012',
                'Tesla Model S Sedan 2012',
                'Toyota 4Runner SUV 2012',
                'Toyota Camry Sedan 2012',
                'Toyota Corolla Sedan 2012',
                'Toyota Sequoia SUV 2012',
                'Volkswagen Beetle Hatchback 2012',
                'Volkswagen Golf Hatchback 1991',
                'Volkswagen Golf Hatchback 2012',
                'Volvo 240 Sedan 1993',
                'Volvo C30 Hatchback 2012',
                'Volvo XC90 SUV 2007'
            ]
        },
        'voc2007': {
            'num_classes': 20,
            'task': 'multi_label_classification',
            'multi_label': True,
            'type': 'file_based',
            'class_names': [
                'aeroplane',
                'bicycle',
                'bird',
                'boat',
                'bottle',
                'bus',
                'car',
                'cat',
                'chair',
                'cow',
                'diningtable',
                'dog',
                'horse',
                'motorbike',
                'person',
                'pottedplant',
                'sheep',
                'sofa',
                'train',
                'tvmonitor'
            ]
        }
    }
    
    def __init__(self, dataset_name: str, root: Optional[str] = None,
                 train: bool = True, download: bool = True,
                 metadata_path: Optional[str] = None):
        """
        Initialize ELEVATER dataset handler.
        
        Args:
            dataset_name: Name of the ELEVATER dataset
            root: Root directory for dataset storage
            train: Whether to load training set (for torchvision datasets)
            download: Whether to download dataset if not found (for torchvision datasets)
            metadata_path: Path to metadata JSON file
        """
        super().__init__()
        
        if dataset_name not in self.ELEVATER_DATASETS:
            raise DatasetError(
                f"Dataset '{dataset_name}' not in ELEVATER. "
                f"Available: {', '.join(sorted(self.ELEVATER_DATASETS.keys()))}"
            )
        
        self.dataset_name = dataset_name
        self.dataset_info = self.ELEVATER_DATASETS[dataset_name]
        self.root = root or './data'
        self.train = train
        self.download = download
        self.data_dir = Path(self.root) / 'elevater'
        self.metadata_path = metadata_path
        self._metadata = None
        self._class_names = None
        self._dataset = None
        
        # Load dataset based on type
        self._load_dataset()
        
        # Load metadata if available
        if metadata_path and Path(metadata_path).exists():
            self._load_metadata()
    
    def _load_dataset(self):
        """Load the dataset based on its type."""
        dataset_type = self.dataset_info.get('type', 'file_based')
        
        if dataset_type == 'torchvision':
            self._load_torchvision_dataset()
        else:
            # For file-based datasets, we don't preload the data
            # Reference samples will be loaded on demand
            pass
    
    def _load_torchvision_dataset(self):
        """Load datasets using torchvision."""
        try:
            import torchvision
            from PIL import Image

            # Custom transform to convert dataset images to PIL (matching original code)
            class ConvertToPIL:
                def __call__(self, img):
                    if not isinstance(img, Image.Image):
                        return Image.fromarray(img)
                    return img

            convert_transform = ConvertToPIL()

            # Add informative message if dataset needs to be downloaded
            if self.download:
                dataset_path = Path(self.root) / self.dataset_name
                if not dataset_path.exists() or not any(dataset_path.iterdir()):
                    print(f"⬇️  Dataset '{self.dataset_name}' not found locally. Downloading...")
                    print(f"   This may take several minutes depending on dataset size.")
                    print(f"   Download location: {self.root}")

            # Define dataset constructors
            dataset_loaders = {
                'cifar10': lambda: torchvision.datasets.CIFAR10(
                    root=self.root, train=self.train, download=self.download, transform=convert_transform),
                'cifar100': lambda: torchvision.datasets.CIFAR100(
                    root=self.root, train=self.train, download=self.download, transform=convert_transform),
                'caltech101': lambda: torchvision.datasets.Caltech101(
                    root=self.root, download=self.download, transform=convert_transform),
                'country211': lambda: torchvision.datasets.Country211(
                    root=self.root, split='train' if self.train else 'test', download=self.download, transform=convert_transform),
                'eurosat': lambda: torchvision.datasets.EuroSAT(
                    root=self.root, download=self.download, transform=convert_transform),
                'food101': lambda: torchvision.datasets.Food101(
                    root=self.root, split='train' if self.train else 'test', download=self.download, transform=convert_transform),
                'gtsrb': lambda: torchvision.datasets.GTSRB(
                    root=self.root, split='train' if self.train else 'test', download=self.download, transform=convert_transform),
                'oxford_flowers102': lambda: torchvision.datasets.Flowers102(
                    root=self.root, split='train' if self.train else 'test', download=self.download, transform=convert_transform),
                'oxford_pets': lambda: torchvision.datasets.OxfordIIITPet(
                    root=self.root, split='trainval' if self.train else 'test', download=self.download, transform=convert_transform),
                'mnist': lambda: torchvision.datasets.MNIST(
                    root=self.root, train=self.train, download=self.download, transform=convert_transform),
                'dtd': lambda: torchvision.datasets.DTD(
                    root=self.root, split='train' if self.train else 'test', download=self.download, transform=convert_transform),
                'fgvc_aircraft': lambda: torchvision.datasets.FGVCAircraft(
                    root=self.root, split='trainval' if self.train else 'test', download=self.download, transform=convert_transform),
            }

            if self.dataset_name not in dataset_loaders:
                raise DatasetError(f"Torchvision support not implemented for {self.dataset_name}")

            self.log_info(f"Initializing {self.dataset_name.upper()} dataset...")
            try:
                self._dataset = dataset_loaders[self.dataset_name]()
                self.log_info(f"Loaded {self.dataset_name.upper()} dataset with {len(self._dataset)} samples")
            except (FileNotFoundError, RuntimeError) as e:
                # Handle Country211's missing class files (SM, SN, SO, SS)
                if self.dataset_name == 'country211' and 'Found no valid file for the classes' in str(e):
                    self.log_warning(f"Some Country211 classes have missing images: {e}")
                    self.log_info("Loading Country211 with available classes only (skipping missing ones)...")
                    # Load without validation - torchvision will skip missing classes
                    import torchvision
                    from torchvision.datasets.folder import ImageFolder

                    # Create custom loader that skips missing classes
                    country211_path = os.path.join(self.root, 'country211', 'train' if self.train else 'test')
                    self._dataset = ImageFolder(root=country211_path, transform=convert_transform)

                    available_classes = len(self._dataset.classes)
                    self.log_info(f"Loaded Country211 with {available_classes}/211 countries ({len(self._dataset)} samples)")
                    self.log_warning(f"Note: {211 - available_classes} countries skipped due to missing images")
                else:
                    raise

        except ImportError:
            raise DatasetError("torchvision is required for torchvision datasets")
        except urllib.error.HTTPError as e:
            # Handle broken download URLs (common with academic datasets)
            if e.code == 404:
                # Torchvision expects: {root}/{dataset_name}/
                expected_structure = (
                    f"{self.root}/{self.dataset_name}/101_ObjectCategories/\n"
                    f"                             ├── accordion/\n"
                    f"                             ├── airplanes/\n"
                    f"                             └── ... (102 class directories)"
                )
                error_msg = (
                    f"Failed to download {self.dataset_name} dataset: Original download URL is broken (404 Not Found).\n\n"
                    f"📥 Manual Download & Extract:\n"
                    f"   wget http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz\n"
                    f"   mkdir -p {self.root}/{self.dataset_name}\n"
                    f"   tar -xzf 101_ObjectCategories.tar.gz -C {self.root}/{self.dataset_name}/\n\n"
                    f"📁 Expected structure:\n{expected_structure}\n\n"
                    f"✅ Alternative: Use a working dataset (cifar10, cifar100, food101, mnist)\n\n"
                    f"💡 After manual download, run the script again - torchvision will detect existing data."
                )
                raise DatasetError(error_msg)
            else:
                raise DatasetError(f"HTTP error downloading {self.dataset_name}: {e}")
        except Exception as e:
            # Check if it's a connection/download error
            error_str = str(e).lower()
            if 'http' in error_str or '404' in error_str or 'url' in error_str or 'download' in error_str:
                error_msg = (
                    f"Failed to download {self.dataset_name} dataset: {e}\n\n"
                    f"This is likely due to broken download URLs in torchvision.\n"
                    f"Please download the dataset manually and place it in: {self.data_dir}/\n\n"
                    f"Alternative: Use a different ELEVATER dataset (cifar10, cifar100, mnist, food101)"
                )
                raise DatasetError(error_msg)
            else:
                raise DatasetError(f"Failed to load {self.dataset_name} dataset: {e}")
    
    def _load_metadata(self):
        """Load dataset metadata from JSON file."""
        try:
            with open(self.metadata_path, 'r') as f:
                self._metadata = json.load(f)
            
            # Extract class names from metadata
            if 'classes' in self._metadata:
                self._class_names = self._metadata['classes']
            elif 'classnames' in self._metadata:
                self._class_names = self._metadata['classnames']
                
            self.log_info(f"Loaded metadata for {self.dataset_name}")
        except Exception as e:
            raise DatasetError(f"Failed to load metadata: {e}")
    
    @property
    def name(self) -> str:
        """Return the dataset name."""
        return self.dataset_name
    
    @property
    def class_names(self) -> List[str]:
        """Return list of class names in the dataset."""
        if self._class_names is not None:
            return self._class_names
        
        # Check if class names are defined in dataset info
        if 'class_names' in self.dataset_info:
            self._class_names = self.dataset_info['class_names']
            return self._class_names
        
        # For metadata-loaded class names
        if self._metadata and 'classes' in self._metadata:
            self._class_names = self._metadata['classes']
            return self._class_names
        elif self._metadata and 'classnames' in self._metadata:
            self._class_names = self._metadata['classnames']
            return self._class_names
        
        # Fallback: try to load from standard locations
        class_file = self.data_dir / self.dataset_name / 'classes.txt'
        if class_file.exists():
            with open(class_file, 'r') as f:
                self._class_names = [line.strip() for line in f]
            return self._class_names
        
        # If no class names found, generate generic ones
        num_classes = self.dataset_info['num_classes']
        self._class_names = [f"class_{i}" for i in range(num_classes)]
        return self._class_names
    
    def get_reference_samples(self, n_per_class: int, seed: int = 42) -> Dict[str, List[Image.Image]]:
        """
        Get reference samples for each class.
        
        Args:
            n_per_class: Number of reference samples per class
            seed: Random seed for reproducible sampling
        """
        # Set random seed for reproducible sampling (both random and numpy)
        random.seed(seed)
        np.random.seed(seed)
        reference_samples = {}
        
        dataset_type = self.dataset_info.get('type', 'file_based')
        
        if dataset_type == 'torchvision':
            # Handle CIFAR datasets using torchvision
            reference_samples = self._get_torchvision_reference_samples(n_per_class)
        else:
            # Handle file-based datasets
            reference_samples = self._get_file_based_reference_samples(n_per_class)
        
        return reference_samples
    
    def _get_torchvision_reference_samples(self, n_per_class: int) -> Dict[str, List[Image.Image]]:
        """Get reference samples from torchvision datasets (CIFAR)."""
        if self._dataset is None:
            raise DatasetError("Dataset not loaded")

        reference_samples = {}

        print(f"Selecting {self.dataset_name.upper()} sample data randomly...")
        print(f"  Dataset size: {len(self._dataset):,} samples")
        print(f"  Number of classes: {len(self.class_names)}")
        print(f"  Samples per class: {n_per_class}")

        # CRITICAL OPTIMIZATION: Build class index map in ONE pass through dataset
        # Instead of iterating 101 times through 75K samples (7.6M iterations),
        # we iterate once through 75K samples (75K iterations)
        print(f"  Building class index map (one-time scan)...")
        class_indices_map = {class_idx: [] for class_idx in range(len(self.class_names))}
        print(f"  Class indices map builded.")
        dataset_size = len(self._dataset)
        progress_interval = max(1, dataset_size // 20)  # Show progress 20 times

        for i in range(dataset_size):
            try:
                _, label = self._dataset[i]
                class_indices_map[label].append(i)

                # Progress indicator
                if (i + 1) % progress_interval == 0 or i == dataset_size - 1:
                    percent = ((i + 1) / dataset_size) * 100
                    print(f"    Progress: {i+1:,}/{dataset_size:,} ({percent:.1f}%)")
            except Exception as e:
                # Skip corrupted samples
                continue

        print(f"  ✅ Index map built. Processing classes...")

        # Now process each class using pre-built index map (much faster!)
        for class_idx, class_name in enumerate(self.class_names):
            # Extract canonical name for dictionary key (FER2013 uses lists)
            canonical_name = class_name[0] if isinstance(class_name, list) else class_name

            class_indices = class_indices_map.get(class_idx, [])

            if not class_indices:
                print(f"    ⚠️  No samples found for class '{canonical_name}'")
                reference_samples[canonical_name] = []
                continue

            print(f"  [{class_idx+1}/{len(self.class_names)}] Class '{canonical_name}': {len(class_indices)} samples")

            # Use np.random.choice exactly like the original code
            sampled_indices = np.random.choice(
                class_indices,
                size=min(n_per_class, len(class_indices)),
                replace=False
            )

            # Get images for this class
            images = []
            for idx in sampled_indices:
                try:
                    # Get the raw image
                    image, _ = self._dataset[idx]

                    # Convert to PIL Image if needed
                    if not isinstance(image, Image.Image):
                        image = Image.fromarray(image)

                    images.append(image)

                except Exception as e:
                    print(f"    ⚠️  Failed to load image at index {idx}: {e}")
                    continue

            reference_samples[canonical_name] = images

        print(f"✅ Reference sampling complete: {len(reference_samples)} classes, {sum(len(imgs) for imgs in reference_samples.values())} total images")
        return reference_samples
    
    def _get_file_based_reference_samples(self, n_per_class: int) -> Dict[str, List[Image.Image]]:
        """Get reference samples from file-based datasets."""
        reference_samples = {}

        # Check if dataset directory exists
        dataset_dir = self.data_dir / self.dataset_name
        if not dataset_dir.exists():
            print(f"❌ Dataset directory not found: {dataset_dir}")
            self.log_warning(
                f"Dataset directory {dataset_dir} not found. "
                "Please download and set up the dataset first."
            )
            return reference_samples

        print(f"📁 Searching for reference images in: {dataset_dir}")

        # Try different common directory structures
        for split in ['train', 'training', 'val', 'validation']:
            split_dir = dataset_dir / split
            if split_dir.exists():
                print(f"  ✓ Found split directory: {split}")

                # List what's actually in the split directory for debugging
                subdirs = [d.name for d in split_dir.iterdir() if d.is_dir()]
                print(f"    Available subdirectories: {subdirs[:10]}")  # Show first 10

                # Look for class subdirectories
                found_classes = 0
                for class_name in self.class_names:
                    # Extract canonical name for dictionary key (FER2013 uses lists)
                    canonical_name = class_name[0] if isinstance(class_name, list) else class_name

                    # Try sanitized class name first (for datasets with problematic characters)
                    sanitized_name = sanitize_filename(canonical_name)
                    class_dir = split_dir / sanitized_name

                    # If sanitized version doesn't exist, try original class name
                    if not class_dir.exists():
                        class_dir = split_dir / canonical_name

                    if class_dir.exists() and class_dir.is_dir():
                        # Get image files
                        image_files = list(class_dir.glob('*.jpg')) + \
                                    list(class_dir.glob('*.png')) + \
                                    list(class_dir.glob('*.jpeg'))

                        if len(image_files) == 0:
                            print(f"    ⚠️  Class '{canonical_name}': directory exists but no images found")
                            continue

                        # Sample images using np.random.choice for consistency
                        sampled_indices = np.random.choice(
                            len(image_files),
                            size=min(n_per_class, len(image_files)),
                            replace=False
                        )
                        sampled_files = [image_files[i] for i in sampled_indices]

                        # Load images
                        images = []
                        for img_file in sampled_files:
                            try:
                                img = Image.open(img_file).convert('RGB')
                                images.append(img)
                            except Exception as e:
                                self.log_warning(f"Failed to load {img_file}: {e}")

                        if images:
                            reference_samples[canonical_name] = images
                            found_classes += 1
                            print(f"    ✓ Class '{canonical_name}': loaded {len(images)} images")
                    else:
                        print(f"    ✗ Class '{canonical_name}': directory not found at {class_dir}")

                print(f"  Summary: Found {found_classes}/{len(self.class_names)} classes with images")

                if reference_samples:
                    break

        if not reference_samples:
            print(f"❌ No reference samples found for {self.dataset_name}")
            print(f"   Looked in: {dataset_dir}")
            print(f"   Expected class names: {self.class_names}")

        return reference_samples
    
    def get_dataloader(self, batch_size: int = 32, shuffle: bool = True, 
                      num_workers: int = 0) -> DataLoader:
        """
        Get a PyTorch DataLoader for torchvision datasets.
        
        Args:
            batch_size: Batch size for loading
            shuffle: Whether to shuffle the data
            num_workers: Number of worker processes
            
        Returns:
            PyTorch DataLoader
        """
        if self.dataset_info.get('type') != 'torchvision':
            raise DatasetError(f"DataLoader only supported for torchvision datasets, not {self.dataset_name}")
        
        if self._dataset is None:
            raise DatasetError("Dataset not loaded")
        
        # Custom transform to ensure PIL Image output
        def pil_transform(img):
            if not isinstance(img, Image.Image):
                return Image.fromarray(img)
            return img
        
        # Apply transform
        original_transform = self._dataset.transform
        self._dataset.transform = pil_transform
        
        dataloader = DataLoader(
            self._dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self._custom_collate
        )
        
        # Restore original transform
        self._dataset.transform = original_transform
        
        return dataloader
    
    def _custom_collate(self, batch):
        """Custom collate function to handle PIL images."""
        images = [item[0] for item in batch]
        labels = torch.tensor([item[1] for item in batch])
        return images, labels
    
    def get_text_templates(self) -> List[str]:
        """
        Get text templates for creating prompts.
        
        Templates can be customized based on the specific ELEVATER dataset.
        """
        # Dataset-specific templates
        dataset_templates = {
            'dtd': [
                "a photo of a {} texture.",
                "a photo of a {} pattern.",
                "a photo of a {} thing.",
                "a photo of a {} object.",
                "a photo of the {} texture.",
                "a photo of the {} pattern.",
                "a photo of the {} thing.",
                "a photo of the {} object."
            ],
            'eurosat': [
                "a centered satellite photo of {}.",
                "a centered satellite photo of a {}.",
                "a centered satellite photo of the {}."
            ],
            'fer2013': [
                "a photo of a {} looking face.",
                "a photo of a face showing the emotion: {}.",
                "a photo of a face looking {}.",
                "a face that looks {}.",
                "they look {}.",
                "look at how {} they are."
            ],
            'food101': [
                "a photo of {}, a type of food"
            ],
            'gtsrb': [
                "a zoomed in photo of a {} traffic sign.",
                "a centered photo of a {} traffic sign.",
                "a close up photo of a {} traffic sign."
            ],
            'oxford_flowers102': [
                "a photo of a {}, a type of flower",
                "a photo of the {}, a type of flower"
            ],
            'oxford_pets': [
                "a photo of a {}, a type of pet.",
                "a photo of the {}, a type of pet."
            ],
            'cifar10': [
                "a photo of a {}.",
                "a blurry photo of a {}.",
                "a black and white photo of a {}.",
                "a low contrast photo of a {}.",
                "a high contrast photo of a {}.",
                "a bad photo of a {}.",
                "a good photo of a {}.",
                "a photo of a small {}.",
                "a photo of a big {}.",
                "a photo of the {}.",
                "a blurry photo of the {}.",
                "a black and white photo of the {}.",
                "a low contrast photo of the {}.",
                "a high contrast photo of the {}.",
                "a bad photo of the {}.",
                "a good photo of the {}.",
                "a photo of the small {}.",
                "a photo of the big {}."
            ],
            'cifar100': [
                "a photo of a {}.",
                "a blurry photo of a {}.",
                "a black and white photo of a {}.",
                "a low contrast photo of a {}.",
                "a high contrast photo of a {}.",
                "a bad photo of a {}.",
                "a good photo of a {}.",
                "a photo of a small {}.",
                "a photo of a big {}.",
                "a photo of the {}.",
                "a blurry photo of the {}.",
                "a black and white photo of the {}.",
                "a low contrast photo of the {}.",
                "a high contrast photo of the {}.",
                "a bad photo of the {}.",
                "a good photo of the {}.",
                "a photo of the small {}.",
                "a photo of the big {}."
            ],
            'caltech101': [
                "a photo of a {}.",
                "a painting of a {}.",
                "a plastic {}.",
                "a sculpture of a {}.",
                "a sketch of a {}.",
                "a tattoo of a {}.",
                "a toy {}.",
                "a rendition of a {}.",
                "a embroidered {}.",
                "a cartoon {}.",
                "a {} in a video game.",
                "a plushie {}.",
                "a origami {}.",
                "art of a {}.",
                "graffiti of a {}.",
                "a drawing of a {}.",
                "a doodle of a {}.",
                "a photo of the {}.",
                "a painting of the {}.",
                "the plastic {}.",
                "a sculpture of the {}.",
                "a sketch of the {}.",
                "a tattoo of the {}.",
                "the toy {}.",
                "a rendition of the {}.",
                "the embroidered {}.",
                "the cartoon {}.",
                "the {} in a video game.",
                "the plushie {}.",
                "the origami {}.",
                "art of the {}.",
                "graffiti of the {}.",
                "a drawing of the {}.",
                "a doodle of the {}."
            ],
            'country211': [
                "a photo i took in {}.",
                "a photo i took while visiting {}.",
                "a photo from my home country of {}.",
                "a photo from my visit to {}.",
                "a photo showing the country of {}."
            ],
            'fgvc_aircraft': [
                "a photo of a {}, a type of aircraft",
                "a photo of the {}, a type of aircraft"
            ],
            'hateful_memes': [
                "a {}.",
                "the {}."
            ],
            'kitti_distance': [
                "{}"
            ],
            'mnist': [
                "a photo of the number: {}."
            ],
            'patchcamelyon': [
                "this is a photo of {}"
            ],
            'rendered_sst2': [
                "a {} review of a movie."
            ],
            'resisc45': [
                "satellite imagery of {}.",
                "aerial imagery of {}.",
                "satellite photo of {}.",
                "aerial photo of {}.",
                "satellite view of {}.",
                "aerial view of {}.",
                "satellite imagery of a {}.",
                "aerial imagery of a {}.",
                "satellite photo of a {}.",
                "aerial photo of a {}.",
                "satellite view of a {}.",
                "aerial view of a {}.",
                "satellite imagery of the {}.",
                "aerial imagery of the {}.",
                "satellite photo of the {}.",
                "aerial photo of the {}.",
                "satellite view of the {}.",
                "aerial view of the {}."
            ],
            'stanford_cars': [
                "a photo of a {}.",
                "a photo of the {}.",
                "a photo of my {}.",
                "i love my {}!",
                "a photo of my dirty {}.",
                "a photo of my clean {}.",
                "a photo of my new {}.",
                "a photo of my old {}."
            ],
            'voc2007': [
                "a photo of a {}."
            ]
            
        }
        
        # Return dataset-specific templates if available
        if self.dataset_name in dataset_templates:
            return dataset_templates[self.dataset_name]
        
        # Default templates
        return [
            "a photo of a {}",
            "a clear image of a {}",
            "a picture showing a {}",
            "a photograph of a {}",
            "an image of a {}",
            "a high-quality photo of a {}",
            "a detailed view of a {}"
        ]
