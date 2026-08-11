#!/usr/bin/env python3
"""
Expand the OOD test set: 403 → 1,000+ rows.

Why this matters
----------------
At 403 rows the 95% confidence interval is ±8.5 pts.  That is wider than the
effect size of almost every experiment in this project — teacher swap, capacity,
vocabulary.  Growing to 1,000+ rows cuts the floor to ±5.4 pts; 2,000 rows gets
it to ±3.8 pts.  Steps 4 and 5 (typo robustness, volume hard-negatives) cannot be
read reliably on the OOD column until this is done.

Content requirements
--------------------
- All rows are GENUINELY out-of-domain for a hearing-aid voice assistant:
  they refer to things the device cannot do (weather, news, shopping, ...).
- No row should be a rephrasing of anything that IS in the hearing-aid domain
  (volume, memory, mute, streaming, phone-find).
- Rows must not duplicate existing ood_test_en.csv content.
- All labels are 'Default Fallback Intent', as the original file.

Usage
-----
    python scripts/expand_ood_testset.py              # append + write
    python scripts/expand_ood_testset.py --dry-run    # print stats, no write
    python scripts/expand_ood_testset.py --out /tmp/ood_preview.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from scripts.common import token_key  # noqa: E402

FALLBACK = "Default Fallback Intent"

# ---------------------------------------------------------------------------
# New OOD phrases — 13 semantic categories, ~50 each = ~650 total
# Every phrase must be something a hearing-aid assistant CANNOT handle.
# ---------------------------------------------------------------------------

NEW_OOD_PHRASES: list[str] = [

    # Smart home / IoT (non-hearing-aid)
    "turn off the living room lights",
    "dim the kitchen lights to fifty percent",
    "set the thermostat to twenty two degrees",
    "lock the front door",
    "unlock the back door remotely",
    "turn on the garden sprinklers",
    "is the garage door closed",
    "start the robot vacuum",
    "pause the robot vacuum",
    "what is the temperature in the living room",
    "turn on the porch light",
    "set a bedtime routine for nine pm",
    "show me the front door camera",
    "close the garage door",
    "set the fan speed to medium",
    "is the dishwasher running",
    "start the washing machine",
    "turn on the coffee maker",
    "schedule the lawn mower for saturday morning",
    "lower the blinds in the bedroom",
    "open the window shades",
    "is the baby monitor on",
    "connect the speaker to bluetooth",
    "turn on the christmas tree lights",
    "set a mood lighting scene",
    "turn off all the lights in the house",
    "what devices are connected to my wifi",
    "check the energy usage this week",
    "show me the backyard camera",
    "is the stove still on",
    "open the front gate",
    "activate vacation mode",
    "disable the motion sensors",
    "set the water heater to eco mode",
    "start the air purifier",
    "turn on night mode",
    "how much battery does my vacuum have",
    "add dishwasher salt to my shopping list",
    "set the indoor humidity to forty five percent",
    "is the basement window open",
    "turn off the fireplace",
    "activate do not disturb mode on all devices",
    "what is my home electricity bill this month",
    "check if my package was delivered",
    "unlock the car remotely",
    "is the car locked",
    "show me the driveway camera feed",
    "set the pool temperature to twenty eight degrees",

    # Shopping / e-commerce
    "order more paper towels",
    "add cereal to my cart",
    "track my amazon order",
    "what is the return policy for this item",
    "find me a deal on running shoes",
    "compare prices for the new iphone",
    "buy more coffee pods",
    "search for black friday deals",
    "what coupons are available today",
    "reorder my prescription",
    "find me organic apples near me",
    "check if the blue jacket is in stock",
    "what is the best price for a dishwasher",
    "place my usual grocery order",
    "how long until my delivery arrives",
    "cancel my most recent order",
    "find me a gift for my sister's birthday",
    "is there a discount on gym memberships",
    "order a large pepperoni pizza",
    "add eggs and bread to my shopping list",
    "search for wireless headphones under fifty dollars",
    "find the best laptop deals right now",
    "remind me to buy milk when i pass a shop",
    "track my fedex shipment",
    "what is the cheapest flight to new york",
    "look up the ingredients for beef stew",
    "find me a plumber near me",
    "is the pharmacy open right now",
    "what are the store hours for ikea",
    "order flowers for delivery tomorrow",
    "find me a good mechanic nearby",
    "get me a taxi to the airport",
    "book an uber to downtown",
    "what restaurants are open for dinner",
    "order sushi for delivery",
    "find me a dog groomer",
    "book a table for four at an italian restaurant",
    "find me a babysitter for friday night",
    "search for a handyman to fix my fence",
    "is the supermarket nearby open now",
    "add toothpaste to my next delivery",
    "cancel my food delivery order",
    "what is the menu at the local burger place",
    "find the nearest petrol station",
    "look for a kids birthday party venue",
    "find me a hotel in lisbon for next week",
    "check flight availability for next tuesday",
    "what is the best airline for flying to tokyo",
    "book a train ticket to edinburgh",
    "find me a car rental for the weekend",

    # Navigation / Maps
    "navigate to the nearest hospital",
    "what is the fastest route to work",
    "how long will it take to drive downtown",
    "is there traffic on the motorway",
    "find a parking spot near the stadium",
    "take me to the airport",
    "show me the scenic route to the coast",
    "avoid motorways on my route",
    "how far is the nearest pharmacy",
    "where is the nearest atm",
    "what is the public transport route to the museum",
    "find the nearest charging station for electric cars",
    "how do i get to the shopping centre from here",
    "show me cycle paths in the city",
    "is the road flooded on the main street",
    "find the nearest petrol station that sells diesel",
    "where can i park for free downtown",
    "is there a bike rental nearby",
    "navigate home avoiding the school zone",
    "what bus goes to the city centre",
    "show me nearby coffee shops",
    "how many miles to the next service station",
    "is the tunnel open today",
    "what is the toll for this route",
    "show me a map of the national park",
    "find a walking trail near here",
    "how do i get to the nearest swimming pool",
    "route me to the vet",
    "find the nearest wheelchair accessible restaurant",
    "how do i get to heathrow terminal five",
    "show me train times from london to paris",
    "what time is the next bus to the station",
    "find me a shortcut to avoid construction",
    "is the level crossing open",
    "navigate back to where i parked",
    "where is the nearest post office",
    "find the quickest route home avoiding school run",
    "how far is it to walk to the beach",
    "show me a satellite view of my house",
    "is there a speed camera ahead",
    "what is the speed limit on this road",
    "how do i get to the nearest urgent care clinic",
    "find a dog-friendly pub nearby",
    "show me the nearest defibrillator",
    "navigate to the office via the scenic road",
    "how long does the ferry to the island take",
    "when does the next tram arrive",
    "find me a taxi rank nearby",
    "is the airport road congested right now",

    # Social / Messaging / Calls
    "send a text to my wife saying i will be late",
    "call my brother",
    "read my messages",
    "reply to the last text i received",
    "call an ambulance",
    "set up a video call with grandma",
    "send an email to my boss",
    "forward that email to my team",
    "schedule a meeting for tomorrow at noon",
    "add a reminder to call the dentist",
    "post a photo to instagram",
    "check my facebook notifications",
    "send a whatsapp to my friend",
    "leave a voicemail for my colleague",
    "block this unknown number",
    "who called me this morning",
    "redial the last number",
    "set a contact for sarah with her new number",
    "remind me to wish tom happy birthday",
    "how many unread emails do i have",
    "mark all emails as read",
    "unsubscribe from this newsletter",
    "remind me to call the insurance company on monday",
    "join the video meeting",
    "mute my phone for the next hour",
    "turn off notifications until nine am",
    "create a new group chat with my family",
    "read the latest news headline to me",
    "share my location with my wife",
    "find my contacts named david",
    "cancel the two o'clock call",
    "reschedule the meeting to thursday afternoon",
    "read out my calendar for tomorrow",
    "who is attending the friday meeting",
    "send the report to the finance team",
    "check if my flight has been confirmed",
    "send a birthday message to uncle james",
    "tell my mom i am on my way",
    "add new contact claire with mobile number",

    # News / Current events / Facts
    "what is in the news today",
    "read me the headlines",
    "what happened in the stock market today",
    "who won the election",
    "what is the latest on the climate summit",
    "tell me about the us economy",
    "what is today's top story",
    "is there any breaking news",
    "what did the prime minister announce",
    "read me the sports scores",
    "who won the champions league last night",
    "what is happening at the g seven summit",
    "has the ceasefire been agreed",
    "what is the latest iphone model",
    "when was the eiffel tower built",
    "who discovered penicillin",
    "what is the boiling point of water in fahrenheit",
    "how tall is mount everest",
    "what is the capital of australia",
    "how many planets are in the solar system",
    "what is the speed of light",
    "who wrote the da vinci code",
    "what is the population of china",
    "when did world war two end",
    "how old is the universe",
    "who is the current chancellor of germany",
    "what languages are spoken in switzerland",
    "who invented the world wide web",
    "when was the berlin wall torn down",
    "what is the chemical symbol for gold",
    "how many bones are in the human body",
    "who painted the sistine chapel",
    "what is the longest river in africa",
    "who won the oscar for best picture last year",
    "when is the next solar eclipse",
    "what year did the titanic sink",
    "how far is the moon from earth",
    "what is the currency of norway",
    "who discovered america",
    "what is the tallest building in the world",
    "who holds the record for the fastest one hundred metres",
    "what is the deepest lake in the world",
    "when was shakespeare born",
    "what is the largest country by area",
    "how many languages are there in the world",
    "who designed the sydney opera house",
    "when was the first iphone released",
    "what is the national animal of canada",
    "who won the most recent wimbledon",

    # Maths / Calculations
    "what is forty seven times thirteen",
    "calculate the square root of one hundred and forty four",
    "what is twenty percent of three hundred",
    "convert two hundred dollars to euros",
    "what is three hundred and sixty divided by twelve",
    "add up five nine twelve and eight",
    "what is the cube root of twenty seven",
    "how many seconds are in a day",
    "convert ten kilograms to pounds",
    "what is the area of a circle with radius five",
    "convert sixty miles per hour to kilometres",
    "what is two to the power of ten",
    "what is the prime factorisation of sixty",
    "how many centimetres in a foot",
    "what is the hypotenuse of a right triangle with sides three and four",
    "what is the fibonacci sequence to ten terms",
    "what is log base ten of one thousand",
    "convert one hundred fahrenheit to celsius",
    "what is the volume of a sphere with radius three",
    "how many minutes in a week",
    "multiply sixty three by seven",
    "what is the least common multiple of six and eight",
    "what is the greatest common divisor of forty eight and eighteen",
    "how many ounces in a kilogram",
    "what is the mean of five eight twelve and fifteen",
    "solve x squared minus five x plus six equals zero",
    "what is the derivative of x cubed",
    "what is the integral of two x",
    "what is sine of thirty degrees",

    # Health / Medical
    "how many calories are in a banana",
    "what are the symptoms of diabetes",
    "what is a normal blood pressure reading",
    "how do i lower my cholesterol",
    "what foods are high in vitamin d",
    "is it normal to feel tired after eating",
    "what are the side effects of ibuprofen",
    "how do i treat a burn",
    "what is the recommended daily water intake",
    "is coffee bad for your heart",
    "how many steps should i walk a day",
    "what is the best exercise for back pain",
    "how do i relieve a headache without medication",
    "what vitamins should i take for energy",
    "is it safe to take paracetamol every day",
    "how do i stop snoring",
    "what are the early signs of a heart attack",
    "how long does the flu last",
    "how do i get better sleep",
    "what does a high white blood cell count mean",
    "what is the best diet for losing weight",
    "how do i check my blood pressure at home",
    "can stress cause high blood pressure",
    "what is the difference between type one and type two diabetes",
    "how do i stretch my hamstrings",
    "what is a normal resting heart rate",
    "how often should i get a health check",
    "what are signs of vitamin b12 deficiency",
    "how do i treat a sprained ankle",
    "is running bad for your knees",

    # Creativity / Entertainment
    "tell me a joke",
    "write me a haiku about autumn",
    "compose a short poem for my wedding",
    "give me a random fun fact",
    "tell me a bedtime story",
    "recommend a good thriller novel",
    "what are the best movies on netflix right now",
    "play a word game with me",
    "sing me a birthday song",
    "recommend a podcast about history",
    "what is a good recipe for chocolate chip cookies",
    "generate a random story opening",
    "what are some icebreaker games for team meetings",
    "come up with a name for my new cafe",
    "write a funny caption for my photo",
    "recommend a documentary about space",
    "what is a good board game for kids",
    "suggest a creative date night idea",
    "give me five ideas for a birthday gift",
    "what are some fun activities for a rainy day",
    "recommend a ted talk on leadership",
    "write a short motivational quote",
    "tell me something surprising about octopuses",
    "what is the plot of hamlet",
    "name the seven dwarfs",
    "who is the author of harry potter",

    # Productivity / Work
    "draft a meeting agenda for tomorrow",
    "translate this sentence into french",
    "create a spreadsheet template for monthly expenses",
    "help me write a professional email declining a meeting",
    "suggest a title for my presentation",
    "create a project timeline for two weeks",
    "help me brainstorm marketing ideas",
    "what is the best way to run a standup meeting",
    "how do i write an executive summary",
    "translate hello into japanese",
    "how do i use pivot tables in excel",
    "what is the keyboard shortcut for paste in windows",
    "how do i merge cells in excel",
    "help me format this cv",
    "write a professional bio for linkedin",
    "how do i create a formula in google sheets",
    "generate a password with ten characters",
    "how do i compress a pdf file",
    "what is the best free project management tool",
    "how do i add a signature to my email",
    "explain the agile methodology in simple terms",
    "what is the difference between scrum and kanban",
    "how do i make a gantt chart",
    "write a job description for a software engineer",
    "what questions should i ask in a job interview",
    "help me prepare for a performance review",
    "what is the best way to take meeting notes",

    # Calendar / Reminders / Time
    "what day is christmas this year",
    "how many days until new year",
    "set a reminder for tuesday at three pm",
    "what is the date today",
    "what time is it in new york right now",
    "when is easter this year",
    "how many weeks until my holiday",
    "what is the time in tokyo",
    "remind me to take my medication at eight am",
    "set a weekly reminder to do laundry on sundays",
    "when is the next bank holiday",
    "how many days are in february this year",
    "what time does the sun set today",
    "what time does the sun rise tomorrow",
    "how many hours until midnight",
    "when is daylight saving time",
    "add my dentist appointment to the calendar",
    "what is on my agenda for this week",
    "cancel my friday morning appointment",
    "move my three o'clock to four thirty",
    "how many days have passed since the first of january",
    "when is ramadan this year",
    "what day of the week was i born",
    "is next monday a public holiday",
    "schedule a recurring reminder every friday",
    "what time is it in sydney australia",
    "how long until the train leaves",
    "set a twenty minute timer for cooking",

    # Finance / Money
    "what is the current bitcoin price",
    "how is the stock market doing today",
    "what is the exchange rate for dollars to pounds",
    "transfer fifty pounds to my savings account",
    "what is my account balance",
    "pay my electricity bill",
    "how do i open an investment account",
    "what is the best savings account interest rate",
    "how do i file my tax return",
    "what is capital gains tax",
    "how much should i save for retirement",
    "should i invest in index funds or stocks",
    "what is the difference between a roth ira and a traditional ira",
    "how do i build an emergency fund",
    "what credit score do i need to buy a house",
    "how do i improve my credit score",
    "what is the average rent in london",
    "how do i apply for a mortgage",
    "what is the current inflation rate",
    "how do i budget my salary",
    "what is a good return on investment",
    "when is the last day to pay my credit card bill",
    "how do i dispute a bank charge",
    "what is a pension and how does it work",
    "should i pay off my mortgage early",
    "how do i send money internationally",
    "what is the transaction fee for paypal",
    "what happens if i miss a mortgage payment",
    "how do i freeze my credit card",
    "what is the cheapest way to send money abroad",

    # Sports / Fitness
    "who won the premier league last season",
    "when is the next formula one race",
    "how many goals did messi score this season",
    "who is ranked number one in tennis right now",
    "when does the nfl season start",
    "what is the current score of the game",
    "who won the last cricket world cup",
    "how do i improve my running pace",
    "what is a good five kilometre time for a beginner",
    "how do i train for a marathon",
    "what is the fastest one hundred metres time ever",
    "how many sets are in a tennis match",
    "what is the offside rule in football",
    "how do i do a proper squat",
    "what is the best warm up before a workout",
    "how many calories does cycling burn per hour",
    "what muscles does a plank work",
    "how do i improve my swimming technique",
    "what is the difference between aerobic and anaerobic exercise",
    "who is the highest paid footballer in the world",
    "when is the tour de france",
    "who won the most recent super bowl",
    "how many laps is a standard swimming race",
    "what is a personal best in running",
    "how do i do a pull up if i am a beginner",
    "recommend a fifteen minute morning workout",
    "how often should i rest between workouts",
    "what is hiit training",
    "how do i stretch properly after a run",
    "who holds the world record for the marathon",

    # Conversational / Chit-chat
    "what is your purpose",
    "can you keep a secret",
    "are you smarter than a human",
    "do you dream",
    "what is the meaning of life",
    "can you be my friend",
    "what should i have for dinner",
    "make me laugh",
    "are you happy right now",
    "who is your creator",
    "do you know any magic tricks",
    "what would you do if you were human",
    "what do you think about artificial intelligence",
    "if you could travel anywhere where would you go",
    "what is your favourite book",
    "can you lie",
    "what is your biggest fear",
    "do you ever get bored",
    "can you get angry",
    "what do you think of the weather today",
    "what is love",
    "can you give me life advice",
    "what superpower would you choose",
    "how do you say hello in ten different languages",
    "who would win in a fight superman or batman",
    "what is the best movie ever made",
    "what do you think about social media",
    "can you help me feel better",
    "what is a fun fact about elephants",
    "tell me something i do not know",

    # Food & Cooking
    "how do i make pasta carbonara",
    "what is a substitute for butter in baking",
    "how long do i cook chicken breast in the oven",
    "what temperature should i bake a potato at",
    "how do i make sourdough bread from scratch",
    "what wine pairs with salmon",
    "how do i caramelise onions",
    "what is the difference between baking soda and baking powder",
    "how do i make homemade ice cream",
    "what herbs go with lamb",
    "how long can i keep leftovers in the fridge",
    "how do i soften butter quickly",
    "what is the best way to cook steak",
    "how do i make a roux",
    "can i freeze cooked rice",
    "how do i make hollandaise sauce",
    "what is the difference between stock and broth",
    "how do i julienne vegetables",
    "what temperature should beef be cooked to",
    "how do i make whipped cream by hand",
    "what are some easy weeknight dinner ideas",
    "how do i prevent pasta from sticking together",
    "what is a good vegetarian protein source",
    "how do i cook quinoa",
    "how long does homemade jam last",

    # Travel & Holidays
    "do i need a visa to visit japan",
    "what vaccinations do i need for thailand",
    "what is the best time of year to visit iceland",
    "how do i apply for a us visa",
    "what are the baggage restrictions on easyjet",
    "can i bring liquids in my hand luggage",
    "what is the currency in new zealand",
    "how do i get from rome airport to the city centre",
    "what is the best travel insurance",
    "how many days should i spend in paris",
    "what is the cheapest month to fly to australia",
    "do i need travel insurance for europe",
    "how do i get a global entry card",
    "what are the top attractions in barcelona",
    "how do i apply for a working holiday visa",
    "is it safe to drink tap water in mexico",
    "what adaptor do i need for australia",
    "how do i get a refund on a cancelled flight",
    "what is the luggage allowance on ryanair",
    "what are the best hidden gems in europe",
    "how do i find cheap accommodation",
    "what is the best travel credit card",
    "what documents do i need to travel with my child",
    "how early should i arrive at the airport",
    "what is airport fast track security",

    # Parenting & Family
    "when should my baby start solid food",
    "what are the milestones for a one year old",
    "how do i get my toddler to sleep through the night",
    "what are the best educational toys for a three year old",
    "how do i deal with a child's tantrum",
    "when do babies start teething",
    "what age should a child learn to read",
    "how do i potty train my child",
    "what are signs of autism in toddlers",
    "how do i introduce a new sibling to my child",
    "what is a healthy diet for a teenager",
    "how do i talk to my teenager about mental health",
    "what should i do if my child is being bullied",
    "how do i choose a school for my child",
    "what are the best books for young children",
    "how do i childproof my home",
    "what age should a child have a mobile phone",
    "how do i manage screen time for my kids",
    "what are the signs of postpartum depression",
    "how do i cope with parenting stress",
    "what are some family activities for the weekend",
    "how do i deal with sibling rivalry",
    "what vitamins does my child need",
    "how do i teach my child about money",
    "what is the best age to teach swimming",

    # Pets & Animals
    "what should i feed my labrador",
    "how often should i walk my dog",
    "what are signs my cat is sick",
    "how do i train a puppy not to bite",
    "what vaccinations does my dog need",
    "how do i get rid of fleas on my cat",
    "what is the best food for a senior dog",
    "how do i introduce a cat to a new dog",
    "how do i stop my dog from barking",
    "what is the average lifespan of a golden retriever",
    "how do i teach my dog to sit",
    "is chocolate really dangerous for dogs",
    "what plants are toxic to cats",
    "how often should i take my dog to the vet",
    "how do i clean my dog's ears",
    "what fish can i keep with goldfish",
    "how do i care for a hamster",
    "what is the best diet for a rabbit",
    "how do i tame a budgie",
    "can cats eat tuna every day",

    # Home & DIY
    "how do i bleed a radiator",
    "how do i fix a dripping tap",
    "what paint finish should i use in a bathroom",
    "how do i hang wallpaper",
    "what is the best way to remove mould from walls",
    "how do i lay laminate flooring",
    "what tools do i need for basic diy",
    "how do i plaster a wall",
    "how do i unblock a drain",
    "what is the best way to insulate a loft",
    "how do i change a fuse in a plug",
    "how do i tile a bathroom",
    "what is the best exterior paint for wood",
    "how do i fix a squeaky floorboard",
    "how do i rewire a plug",
    "what type of screws should i use for plasterboard",
    "how do i clean gutters safely",
    "how do i install a door handle",
    "what is the best way to seal a bath",
    "how do i stop condensation on windows",

    # Language & Learning
    "how do i learn spanish quickly",
    "what is the best app to learn mandarin",
    "how many words do i need to know to be fluent in french",
    "what is the difference between who and whom",
    "how do i improve my spelling",
    "what is the oxford comma",
    "how do i write a conclusion for an essay",
    "what is the passive voice",
    "how do i learn touch typing",
    "what is the best way to memorise vocabulary",
    "how do i improve my public speaking",
    "what is cognitive load theory",
    "how do i speed read",
    "what is the pomodoro technique",
    "how do i improve my writing style",
    "what does per se mean",
    "how do i write a proper bibliography",
    "what is the difference between i e and e g",
    "how do i cite a website in apa format",
    "what is a dangling modifier",

    # Environment & Science
    "what causes climate change",
    "how do i reduce my carbon footprint",
    "what is the difference between weather and climate",
    "how do solar panels work",
    "what is net zero",
    "what is a heat pump and how does it work",
    "how do i compost at home",
    "what is biodegradable packaging",
    "how do electric cars work",
    "what is the greenhouse effect",
    "what are the benefits of going vegan for the environment",
    "how do i recycle correctly",
    "what is fast fashion and why is it bad",
    "how do i save water at home",
    "what is a carbon offset",
    "how does wind energy work",
    "what is a smart meter",
    "how much of the ocean is unexplored",
    "what is the difference between a hurricane and a tornado",
    "how do volcanoes form",
]


def main() -> int:
    ap = argparse.ArgumentParser(description="Expand ood_test_en.csv to 1,000+ rows")
    ap.add_argument("--out", type=Path, default=config.DATA / "eval" / "ood_test_en.csv",
                    help="output path (default: same as existing OOD file -- appends in-place)")
    ap.add_argument("--dry-run", action="store_true", help="print stats without writing")
    args = ap.parse_args()

    ood_path = config.DATA / "eval" / "ood_test_en.csv"

    # load existing rows (preserve order, exact dedup)
    existing: list[tuple[str, str]] = []
    if ood_path.exists():
        with open(ood_path, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                existing.append((row["text"], row.get("expected", FALLBACK)))

    existing_keys: set[str] = {token_key(t) for t, _ in existing}
    existing_raw: set[str] = {t.strip().lower() for t, _ in existing}

    # filter new phrases
    added: list[tuple[str, str]] = []
    skipped_dup: list[str] = []

    for phrase in NEW_OOD_PHRASES:
        phrase = phrase.strip()
        if not phrase:
            continue
        if phrase.lower() in existing_raw:
            skipped_dup.append(phrase)
            continue
        if token_key(phrase) in existing_keys:
            skipped_dup.append(phrase)
            continue
        added.append((phrase, FALLBACK))
        existing_keys.add(token_key(phrase))
        existing_raw.add(phrase.lower())

    total_new = len(existing) + len(added)
    print(f"existing rows : {len(existing)}")
    print(f"new phrases   : {len(NEW_OOD_PHRASES)}")
    print(f"duplicates    : {len(skipped_dup)}")
    print(f"added         : {len(added)}")
    print(f"total after   : {total_new}")

    if total_new < 1000:
        print(f"\nWARNING: total {total_new} < 1,000 -- noise floor will still be > +/-5.4 pts")
    else:
        import math
        ci = 1.96 * math.sqrt(0.5 * 0.5 / total_new) * 100
        print(f"95% CI width  : +/-{ci:.1f} pts  (was +/-8.5 pts at 403 rows)")

    if skipped_dup:
        print(f"\nSkipped duplicates ({len(skipped_dup)}):")
        for d in skipped_dup[:10]:
            print(f"  {d!r}")
        if len(skipped_dup) > 10:
            print(f"  ... and {len(skipped_dup) - 10} more")

    if args.dry_run:
        print("\n(dry run -- nothing written)")
        return 0

    all_rows = existing + added
    dest = args.out
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["text", "expected"])
        w.writerows(all_rows)

    print(f"\nwrote {dest}  ({total_new} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
