"""Support for ELEVATER benchmark datasets including CIFAR-10/100."""

from typing import Dict, List, Optional, Any
import json
import random
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import DataLoader

from ..core.base import BaseDataset
from ..core.exceptions import DatasetError


class ELEVATERDataset(BaseDataset):
    """
    Handler for ELEVATER benchmark datasets.
    
    ELEVATER (Evaluation of Language-augmented Visual Task Adaptation with Eye-Tracking)
    includes multiple vision datasets for comprehensive evaluation.
    """
    
    # Mapping of ELEVATER dataset names to their properties
    ELEVATER_DATASETS = {
        'caltech101': {
            'num_classes': 102,
            'task': 'classification',
            'type': 'torchvision',
            'class_names': [
                'accordion', 'airplanes', 'anchor', 'ant', 'barrel', 'bass', 'beaver', 'binocular',
                'bonsai', 'brain', 'brontosaurus', 'buddha', 'butterfly', 'camera', 'cannon',
                'car_side', 'ceiling_fan', 'cellphone', 'chair', 'chandelier', 'cougar_body',
                'cougar_face', 'crab', 'crayfish', 'crocodile', 'crocodile_head', 'cup', 'dalmatian',
                'dollar_bill', 'dolphin', 'dragonfly', 'electric_guitar', 'elephant', 'emu', 'euphonium',
                'ewer', 'faces', 'faces_easy', 'ferry', 'flamingo', 'flamingo_head', 'garfield',
                'gerenuk', 'gramophone', 'grand_piano', 'hawksbill', 'headphone', 'hedgehog',
                'helicopter', 'ibis', 'inline_skate', 'joshua_tree', 'kangaroo', 'ketch', 'lamp',
                'laptop', 'llama', 'lobster', 'lotus', 'mandolin', 'mayfly', 'menorah', 'metronome',
                'minaret', 'motorbikes', 'nautilus', 'octopus', 'okapi', 'pagoda', 'panda', 'pigeon',
                'pizza', 'platypus', 'pyramid', 'revolver', 'rhino', 'rooster', 'saxophone', 'schooner',
                'scissors', 'scorpion', 'sea_horse', 'snoopy', 'soccer_ball', 'stapler', 'starfish',
                'stegosaurus', 'stop_sign', 'strawberry', 'sunflower', 'tick', 'trilobite', 'umbrella',
                'watch', 'water_lilly', 'wheelchair', 'wild_cat', 'windsor_chair', 'wrench', 'yin_yang',
                'background_google'
            ]
        },
        'cifar10': {
            'num_classes': 10,
            'task': 'classification',
            'type': 'torchvision',
            'class_names': [
                'airplane', 'automobile', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck'
            ]
        },
        'cifar100': {
            'num_classes': 100,
            'task': 'classification',
            'type': 'torchvision',
            'class_names': [
                'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
                'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
                'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
                'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
                'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
                'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
                'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
                'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
                'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
                'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
                'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
                'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
                'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
                'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
            ]
        },
        'country211': {
            'num_classes': 211,
            'task': 'classification',
            'type': 'torchvision',
            'class_names': [
                'andorra', 'united_arab_emirates', 'afghanistan', 'antigua_and_barbuda', 'anguilla',
                'albania', 'armenia', 'angola', 'antarctica', 'argentina', 'american_samoa', 'austria',
                'australia', 'aruba', 'aland_islands', 'azerbaijan', 'bosnia_and_herzegovina', 'barbados',
                'bangladesh', 'belgium', 'burkina_faso', 'bulgaria', 'bahrain', 'burundi', 'benin',
                'saint_barthelemy', 'bermuda', 'brunei', 'bolivia', 'bonaire_sint_eustatius_and_saba',
                'brazil', 'bahamas', 'bhutan', 'bouvet_island', 'botswana', 'belarus', 'belize',
                'canada', 'cocos_islands', 'democratic_republic_of_the_congo', 'central_african_republic',
                'republic_of_the_congo', 'switzerland', 'cote_divoire', 'cook_islands', 'chile',
                'cameroon', 'china', 'colombia', 'costa_rica', 'cuba', 'cape_verde', 'curacao',
                'christmas_island', 'cyprus', 'czech_republic', 'germany', 'djibouti', 'denmark',
                'dominica', 'dominican_republic', 'algeria', 'ecuador', 'estonia', 'egypt',
                'western_sahara', 'eritrea', 'spain', 'ethiopia', 'finland', 'fiji', 'falkland_islands',
                'micronesia', 'faroe_islands', 'france', 'gabon', 'united_kingdom', 'grenada',
                'georgia', 'french_guiana', 'guernsey', 'ghana', 'gibraltar', 'greenland', 'gambia',
                'guinea', 'guadeloupe', 'equatorial_guinea', 'greece', 'south_georgia_and_the_south_sandwich_islands',
                'guatemala', 'guam', 'guinea_bissau', 'guyana', 'hong_kong', 'heard_island_and_mcdonald_islands',
                'honduras', 'croatia', 'haiti', 'hungary', 'indonesia', 'ireland', 'israel', 'isle_of_man',
                'india', 'british_indian_ocean_territory', 'iraq', 'iran', 'iceland', 'italy', 'jersey',
                'jamaica', 'jordan', 'japan', 'kenya', 'kyrgyzstan', 'cambodia', 'kiribati', 'comoros',
                'saint_kitts_and_nevis', 'north_korea', 'south_korea', 'kuwait', 'cayman_islands',
                'kazakhstan', 'laos', 'lebanon', 'saint_lucia', 'liechtenstein', 'sri_lanka', 'liberia',
                'lesotho', 'lithuania', 'luxembourg', 'latvia', 'libya', 'morocco', 'monaco', 'moldova',
                'montenegro', 'saint_martin', 'madagascar', 'marshall_islands', 'macedonia', 'mali',
                'myanmar', 'mongolia', 'macao', 'northern_mariana_islands', 'martinique', 'mauritania',
                'montserrat', 'malta', 'mauritius', 'maldives', 'malawi', 'mexico', 'malaysia',
                'mozambique', 'namibia', 'new_caledonia', 'niger', 'norfolk_island', 'nigeria',
                'nicaragua', 'netherlands', 'norway', 'nepal', 'nauru', 'niue', 'new_zealand', 'oman',
                'panama', 'peru', 'french_polynesia', 'papua_new_guinea', 'philippines', 'pakistan',
                'poland', 'saint_pierre_and_miquelon', 'pitcairn', 'puerto_rico', 'palestine', 'portugal',
                'palau', 'paraguay', 'qatar', 'reunion', 'romania', 'serbia', 'russia', 'rwanda',
                'saudi_arabia', 'solomon_islands', 'seychelles', 'sudan', 'sweden', 'singapore',
                'saint_helena', 'slovenia', 'svalbard_and_jan_mayen', 'slovakia', 'sierra_leone',
                'san_marino', 'senegal', 'somalia', 'suriname', 'south_sudan', 'sao_tome_and_principe',
                'el_salvador', 'sint_maarten', 'syria', 'swaziland', 'turks_and_caicos_islands', 'chad',
                'french_southern_territories', 'togo', 'thailand', 'tajikistan', 'tokelau', 'timor_leste',
                'turkmenistan', 'tunisia', 'tonga', 'turkey', 'trinidad_and_tobago', 'tuvalu', 'taiwan',
                'tanzania', 'ukraine', 'uganda', 'united_states_minor_outlying_islands', 'united_states',
                'uruguay', 'uzbekistan', 'vatican', 'saint_vincent_and_the_grenadines', 'venezuela',
                'british_virgin_islands', 'us_virgin_islands', 'vietnam', 'vanuatu', 'wallis_and_futuna',
                'samoa', 'yemen', 'mayotte', 'south_africa', 'zambia', 'zimbabwe'
            ]
        },
        'dtd': {
            'num_classes': 47,
            'task': 'classification',
            'type': 'file_based'
        },
        'eurosat': {
            'num_classes': 10,
            'task': 'classification',
            'type': 'torchvision',
            'class_names': [
                'annual_crop', 'forest', 'herbaceous_vegetation', 'highway', 'industrial',
                'pasture', 'permanent_crop', 'residential', 'river', 'sea_lake'
            ]
        },
        'fer2013': {
            'num_classes': 7,
            'task': 'classification',
            'type': 'file_based'
        },
        'fgvc_aircraft': {
            'num_classes': 100,
            'task': 'classification',
            'type': 'file_based'
        },
        'food101': {
            'num_classes': 101,
            'task': 'classification',
            'type': 'torchvision',
            'class_names': [
                'apple_pie', 'baby_back_ribs', 'baklava', 'beef_carpaccio', 'beef_tartare', 'beet_salad',
                'beignets', 'bibimbap', 'bread_pudding', 'breakfast_burrito', 'bruschetta', 'caesar_salad',
                'cannoli', 'caprese_salad', 'carrot_cake', 'ceviche', 'cheese_plate', 'cheesecake',
                'chicken_curry', 'chicken_quesadilla', 'chicken_wings', 'chocolate_cake', 'chocolate_mousse',
                'churros', 'clam_chowder', 'club_sandwich', 'crab_cakes', 'creme_brulee', 'croque_madame',
                'cup_cakes', 'deviled_eggs', 'donuts', 'dumplings', 'edamame', 'eggs_benedict',
                'escargots', 'falafel', 'filet_mignon', 'fish_and_chips', 'foie_gras', 'french_fries',
                'french_onion_soup', 'french_toast', 'fried_calamari', 'fried_rice', 'frozen_yogurt',
                'garlic_bread', 'gnocchi', 'greek_salad', 'grilled_cheese_sandwich', 'grilled_salmon',
                'guacamole', 'gyoza', 'hamburger', 'hot_and_sour_soup', 'hot_dog', 'huevos_rancheros',
                'hummus', 'ice_cream', 'lasagna', 'lobster_bisque', 'lobster_roll_sandwich', 'macaroni_and_cheese',
                'macarons', 'miso_soup', 'mussels', 'nachos', 'omelette', 'onion_rings', 'oysters',
                'pad_thai', 'paella', 'pancakes', 'panna_cotta', 'peking_duck', 'pho', 'pizza',
                'pork_chop', 'poutine', 'prime_rib', 'pulled_pork_sandwich', 'ramen', 'ravioli',
                'red_velvet_cake', 'risotto', 'samosa', 'sashimi', 'scallops', 'seaweed_salad',
                'shrimp_and_grits', 'spaghetti_bolognese', 'spaghetti_carbonara', 'spring_rolls',
                'steak', 'strawberry_shortcake', 'sushi', 'tacos', 'takoyaki', 'tiramisu', 'tuna_tartare',
                'waffles'
            ]
        },
        'gtsrb': {
            'num_classes': 43,
            'task': 'classification',
            'type': 'torchvision',
            'class_names': [
                'speed_limit_20', 'speed_limit_30', 'speed_limit_50', 'speed_limit_60', 'speed_limit_70',
                'speed_limit_80', 'end_of_speed_limit_80', 'speed_limit_100', 'speed_limit_120',
                'no_passing', 'no_passing_for_vehicles_over_3.5_metric_tons', 'right_of_way_at_the_next_intersection',
                'priority_road', 'yield', 'stop', 'no_vehicles', 'vehicles_over_3.5_metric_tons_prohibited',
                'no_entry', 'general_caution', 'dangerous_curve_to_the_left', 'dangerous_curve_to_the_right',
                'double_curve', 'bumpy_road', 'slippery_road', 'road_narrows_on_the_right', 'road_work',
                'traffic_signals', 'pedestrians', 'children_crossing', 'bicycles_crossing', 'beware_of_ice_snow',
                'wild_animals_crossing', 'end_of_all_speed_and_passing_limits', 'turn_right_ahead',
                'turn_left_ahead', 'ahead_only', 'go_straight_or_right', 'go_straight_or_left',
                'keep_right', 'keep_left', 'roundabout_mandatory', 'end_of_no_passing',
                'end_of_no_passing_by_vehicles_over_3.5_metric_tons'
            ]
        },
        'hateful_memes': {
            'num_classes': 2,
            'task': 'classification',
            'type': 'file_based'
        },
        'kitti_distance': {
            'num_classes': 4,
            'task': 'classification',
            'type': 'file_based'
        },
        'mnist': {
            'num_classes': 10,
            'task': 'classification',
            'type': 'file_based'
        },
        'oxford_flowers102': {
            'num_classes': 102,
            'task': 'classification',
            'type': 'torchvision',
            'class_names': [
                'pink_primrose', 'hard_leaved_pocket_orchid', 'canterbury_bells', 'sweet_pea', 'wild_geranium',
                'tiger_lily', 'moon_orchid', 'bird_of_paradise', 'monkshood', 'globe_thistle', 'snapdragon',
                'colt_s_foot', 'king_protea', 'spear_thistle', 'yellow_iris', 'globe_flower', 'purple_columbine',
                'peruvian_lily', 'balloon_flower', 'giant_white_arum_lily', 'fire_lily', 'pincushion_flower',
                'fritillary', 'red_ginger', 'grape_hyacinth', 'corn_poppy', 'prince_of_wales_feathers',
                'stemless_gentian', 'artichoke', 'sweet_william', 'carnation', 'garden_phlox', 'love_in_the_mist',
                'cosmos', 'alpine_sea_holly', 'ruby_lipped_cattleya', 'cape_flower', 'great_masterwort',
                'siam_tulip', 'lenten_rose', 'barberton_daisy', 'daffodil', 'sword_lily', 'poinsettia',
                'bolero_deep_blue', 'wallflower', 'marigold', 'buttercup', 'daisy', 'common_dandelion',
                'petunia', 'wild_pansy', 'primula', 'sunflower', 'lilac_hibiscus', 'bishop_of_llandaff',
                'gaura', 'geranium', 'orange_dahlia', 'pink_and_yellow_dahlia', 'cautleya_spicata',
                'japanese_anemone', 'black_eyed_susan', 'silverbush', 'californian_poppy', 'osteospermum',
                'spring_crocus', 'iris', 'windflower', 'tree_poppy', 'gazania', 'azalea', 'water_lily',
                'rose', 'thorn_apple', 'morning_glory', 'passion_flower', 'lotus_lotus', 'toad_lily',
                'anthurium', 'frangipani', 'clematis', 'hibiscus', 'columbine', 'desert_rose', 'tree_mallow',
                'magnolia', 'cyclamen', 'watercress', 'canna_lily', 'hippeastrum', 'bee_balm', 'pink_quill',
                'foxglove', 'bougainvillea', 'camellia', 'mallow', 'mexican_petunia', 'bromelia',
                'blanket_flower', 'trumpet_creeper', 'blackberry_lily', 'common_tulip', 'wild_rose'
            ]
        },
        'oxford_pets': {
            'num_classes': 37,
            'task': 'classification',
            'type': 'torchvision',
            'class_names': [
                'abyssinian', 'american_bulldog', 'american_pit_bull_terrier', 'basset_hound', 'beagle',
                'bengal', 'birman', 'bombay', 'boxer', 'british_shorthair', 'chihuahua', 'egyptian_mau',
                'english_cocker_spaniel', 'english_setter', 'german_shorthaired', 'great_pyrenees',
                'havanese', 'japanese_chin', 'keeshond', 'leonberger', 'maine_coon', 'miniature_pinscher',
                'newfoundland', 'persian', 'pomeranian', 'pug', 'ragdoll', 'russian_blue', 'saint_bernard',
                'samoyed', 'scottish_terrier', 'shiba_inu', 'siamese', 'sphynx', 'staffordshire_bull_terrier',
                'wheaten_terrier', 'yorkshire_terrier'
            ]
        },
        'patchcamelyon': {
            'num_classes': 2,
            'task': 'classification',
            'type': 'file_based'
        },
        'rendered_sst2': {
            'num_classes': 2,
            'task': 'classification',
            'type': 'file_based'
        },
        'resisc45': {
            'num_classes': 45,
            'task': 'classification',
            'type': 'file_based'
        },
        'stanford_cars': {
            'num_classes': 196,
            'task': 'classification',
            'type': 'file_based'
        },
        'voc2007': {
            'num_classes': 20,
            'task': 'classification',
            'type': 'file_based'
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
            }
            
            if self.dataset_name not in dataset_loaders:
                raise DatasetError(f"Torchvision support not implemented for {self.dataset_name}")
            
            self._dataset = dataset_loaders[self.dataset_name]()
            self.log_info(f"Loaded {self.dataset_name.upper()} dataset with {len(self._dataset)} samples")
            
        except ImportError:
            raise DatasetError("torchvision is required for torchvision datasets")
        except Exception as e:
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
        
        # Process each class following the original code pattern exactly
        for class_idx, class_name in enumerate(self.class_names):
            # Find all indices for this class (using class index, not name)
            class_indices = [i for i, (_, label) in enumerate(self._dataset) if label == class_idx]
            
            # Use np.random.choice exactly like the original code
            sampled_indices = np.random.choice(
                class_indices,
                size=min(n_per_class, len(class_indices)),
                replace=False
            )
            
            # Get images for this class
            images = []
            for idx in sampled_indices:
                # Get the raw image
                image, _ = self._dataset[idx]
                
                # Convert to PIL Image if needed
                if not isinstance(image, Image.Image):
                    image = Image.fromarray(image)
                
                images.append(image)
            
            reference_samples[class_name] = images
        
        return reference_samples
    
    def _get_file_based_reference_samples(self, n_per_class: int) -> Dict[str, List[Image.Image]]:
        """Get reference samples from file-based datasets."""
        reference_samples = {}
        
        # Check if dataset directory exists
        dataset_dir = self.data_dir / self.dataset_name
        if not dataset_dir.exists():
            self.log_warning(
                f"Dataset directory {dataset_dir} not found. "
                "Please download and set up the dataset first."
            )
            return reference_samples
        
        # Try different common directory structures
        for split in ['train', 'training', 'val', 'validation']:
            split_dir = dataset_dir / split
            if split_dir.exists():
                # Look for class subdirectories
                for class_name in self.class_names:
                    class_dir = split_dir / class_name
                    if class_dir.exists() and class_dir.is_dir():
                        # Get image files
                        image_files = list(class_dir.glob('*.jpg')) + \
                                    list(class_dir.glob('*.png')) + \
                                    list(class_dir.glob('*.jpeg'))
                        
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
                        
                        reference_samples[class_name] = images
                
                if reference_samples:
                    break
        
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
                "a {} texture",
                "a photo of a {} texture",
                "the {} pattern"
            ],
            'eurosat': [
                "a satellite photo of {}",
                "aerial view of {}",
                "satellite imagery showing {}"
            ],
            'fer2013': [
                "a face showing {}",
                "a person feeling {}",
                "facial expression of {}"
            ],
            'food101': [
                "a photo of {}, a type of food",
                "a dish called {}",
                "food image showing {}"
            ],
            'gtsrb': [
                "a {} traffic sign",
                "road sign showing {}",
                "traffic sign for {}"
            ],
            'oxford_flowers102': [
                "a photo of a {}, a type of flower",
                "a flower called {}",
                "blooming {} flower"
            ],
            'oxford_pets': [
                "a photo of a {}, a type of pet",
                "a pet {}",
                "an animal called {}"
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