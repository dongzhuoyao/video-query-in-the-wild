activitynet_label_list = [
    "Making a lemonade",
    "Painting fence",
    "Playing congas",
    "Archery",
    "Croquet",
    "Doing fencing",
    "Brushing teeth",
    "Pole vault",
    "Playing harmonica",
    "Shaving",
    "Playing guitarra",
    "Washing dishes",
    "Swimming",
    "Drinking beer",
    "Clean and jerk",
    "Kite flying",
    "Throwing darts",
    "Playing piano",
    "Assembling bicycle",
    "Ironing clothes",
    "Playing beach volleyball",
    "Canoeing",
    "Scuba diving",
    "Grooming horse",
    "Removing curlers",
    "Vacuuming floor",
    "Making a cake",
    "Using the balance beam",
    "Using uneven bars",
    "Doing kickboxing",
    "Ping-pong",
    "Doing karate",
    "Camel ride",
    "Fixing the roof",
    "Playing accordion",
    "Long jump",
    "Playing rubik cube",
    "Playing flauta",
    "Playing badminton",
    "Baking cookies",
    "Wrapping presents",
    "Doing a powerbomb",
    "Mooping floor",
    "Waterskiing",
    "Kayaking",
    "Discus throw",
    "Hammer throw",
    "Using the monkey bar",
    "Gargling mouthwash",
    "Changing car wheel",
    "Tango",
    "Playing kickball",
    "Having an ice cream",
    "Elliptical trainer",
    "Cheerleading",
    "Playing polo",
    "Drum corps",
    "Making an omelette",
    "Shaving legs",
    "Slacklining",
    "Getting a haircut",
    "Mixing drinks",
    "Skiing",
    "Longboarding",
    "Rafting",
    "Futsal",
    "Cleaning shoes",
    "Smoking hookah",
    "Fixing bicycle",
    "Playing pool",
    "Cricket",
    "Hopscotch",
    "Horseback riding",
    "Playing saxophone",
    "Calf roping",
    "Hand washing clothes",
    "Disc dog",
    "Shoveling snow",
    "Cleaning windows",
    "Drinking coffee",
    "Mowing the lawn",
    "Washing hands",
    "Polishing forniture",
    "Shuffleboard",
    "Blowing leaves",
    "Putting in contact lenses",
    "Sumo",
    "Using the pommel horse",
    "Putting on shoes",
    "Bathing dog",
    "Decorating the Christmas tree",
    "Doing step aerobics",
    "Walking the dog",
    "Layup drill in basketball",
    "Doing motocross",
    "Washing face",
    "Ballet",
    "Sailing",
    "Braiding hair",
    "River tubing",
    "Starting a campfire",
    "Removing ice from car",
    "Blow-drying hair",
    "Cumbia",
    "Bullfighting",
    "Playing bagpipes",
    "Using parallel bars",
    "Spread mulch",
    "Belly dance",
    "Preparing salad",
    "Preparing pasta",
    "Putting on makeup",
    "Making a sandwich",
    "Fun sliding down",
    "Playing drums",
    "Hurling",
    "Cutting the grass",
    "Surfing",
    "Snowboarding",
    "Peeling potatoes",
    "Beer pong",
    "Swinging at the playground",
    "Windsurfing",
    "Playing ten pins",
    "Tug of war",
    "Playing water polo",
    "High jump",
    "Playing field hockey",
    "Applying sunscreen",
    "Clipping cat claws",
    "Chopping wood",
    "Powerbocking",
    "Snatch",
    "Paintball",
    "Skateboarding",
    "Grooming dog",
    "Raking leaves",
    "Playing violin",
    "Knitting",
    "Using the rowing machine",
    "Capoeira",
    "Tennis serve with ball bouncing",
    "Doing crunches",
    "Laying tile",
    "Roof shingle removal",
    "Rock-paper-scissors",
    "Running a marathon",
    "Rope skipping",
    "Bungee jumping",
    "Carving jack-o-lanterns",
    "Shot put",
    "Polishing shoes",
    "Riding bumper cars",
    "Painting",
    "Getting a piercing",
    "Plastering",
    "Brushing hair",
    "Zumba",
    "Welding",
    "Triple jump",
    "Playing lacrosse",
    "Trimming branches or hedges",
    "Cleaning sink",
    "Playing racquetball",
    "Baton twirling",
    "Rock climbing",
    "Hanging wallpaper",
    "Kneeling",
    "Painting furniture",
    "Tai chi",
    "BMX",
    "Arm wrestling",
    "Breakdancing",
    "Playing squash",
    "Smoking a cigarette",
    "Wakeboarding",
    "Tumbling",
    "Waxing skis",
    "Hand car wash",
    "Springboard diving",
    "Javelin throw",
    "Table soccer",
    "Hula hoop",
    "Sharpening knives",
    "Rollerblading",
    "Dodgeball",
    "Playing blackjack",
    "Curling",
    "Snow tubing",
    "Ice fishing",
    "Building sandcastles",
    "Plataform diving",
    "Doing nails",
    "Installing carpet",
    "Spinning",
    "Beach soccer",
    "Hitting a pinata",
    "Volleyball",
    "Getting a tattoo",
    "Playing ice hockey",
]
# activitynet_label_list = [w.lower() for w in activitynet_label_list]
import getpass

username = getpass.getuser()

import random

random.seed(620)
random.shuffle(activitynet_label_list)
arv_train_label = activitynet_label_list[:80]
arv_val_label = activitynet_label_list[80:100]
arv_test_label = activitynet_label_list[100:]
json_path = (
    "/home/tao/lab/video-query-in-the-wild/data_generate/arv_db_80_20_100.json"
)
json_path = json_path.replace("tao", username)

short_name = "80_20_100"
json_path = "/home/tao/lab/video-query-in-the-wild/data_generate/arv_db_{}.json".format(
    short_name
)
json_path = json_path.replace("tao", username)

moment_eval_json_path = "/home/tao/lab/video-query-in-the-wild/data_generate/arv_db_{}_untrimmed.json".format(
    short_name
)
moment_eval_json_path = moment_eval_json_path.replace("tao", username)
