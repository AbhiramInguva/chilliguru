"""
ChilliGuru v2 🌶️ — Andhra & Telangana Chilli Farming Expert
Setup:
    pip install groq requests transformers torch torchvision pillow
    export GROQ_API_KEY="your-groq-key"
    python3 chilliguru.py
"""

import os
import sys
from pathlib import Path

try:
    import requests
except ImportError:
    print("❌ Missing: pip install requests")
    sys.exit(1)

try:
    from groq import Groq
except ImportError:
    print("❌ Missing: pip install groq")
    sys.exit(1)

import detector

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "YOUR_API_KEY_HERE")
MODEL        = "llama-3.3-70b-versatile"
MAX_TOKENS   = 1600

client = Groq(api_key=GROQ_API_KEY)

SYSTEM_PROMPT = """
You are ChilliGuru, a friendly and helpful farming assistant for chilli farmers in
Andhra Pradesh and Telangana. You talk like a trusted friend who knows a lot about
farming — simple, warm, and easy to understand. No complicated words.

VARIETIES YOU KNOW: Teja, Guntur Sannam, LCA 334, Wonder Hot, Pusa Jwala, Byadgi,
and local varieties across Guntur, Khammam, Warangal, Krishna, Prakasam, Kurnool.

SEASONS:
  Kharif (June-Oct)  -> rainy season, lots of fungal problems and pests
  Rabi   (Nov-Feb)   -> cool season, watch for thrips and powdery mildew
  Zaid   (Mar-May)   -> hot season, spider mites and water stress are common

GROWTH STAGES: seed selection, nursery, transplanting, vegetative, flowering,
fruiting, harvesting, post-harvest storage and drying.

PESTS TO KNOW: Thrips, Spider Mites, Aphids, Whiteflies, Fruit Borer, Mealybugs,
Leaf Miners, Stem Borer.

DISEASES TO KNOW: Leaf Curl Virus, Powdery Mildew, Anthracnose (fruit rot),
Damping Off, Phytophthora, Cercospora Leaf Spot, Bacterial Wilt, Mosaic Virus.

HOW TO WRITE YOUR RESPONSE:
- Use very simple language. Write like you are talking to a farmer face to face.
- No big English words. If you use a technical word, explain it simply in brackets.
- Keep sentences short. One idea per sentence.
- When you give a solution, always include:

  Solution name (Home-made OR From agri shop):
  How to make it: [simple step by step]
  How to use it: [simple instructions]
  How well it works: X out of 10
  How long before you see results: X to X days
  Cost: around Rs X to Rs X
  How often to use: every X days for X weeks
  Where to get it: [local shops or home ingredients in AP/Telangana]

- Always give 2 or 3 solutions - one home-made, one from the shop.
- End with one simple prevention tip for next time.
- If the farmer is a beginner, ask one question at a time to understand their problem.
- If they are experienced, just answer directly.

ONLY USE ORGANIC SOLUTIONS. Never suggest any chemical spray or synthetic fertiliser.

Home-made options: neem leaf water, garlic-chilli spray, cow urine mix, jeevamrutha,
panchagavya, tobacco water, soap water, turmeric water, buttermilk spray.

Shop-bought organic options: Neemazal or Econeem (neem oil), Multiplex Trichoderma,
Multiplex Pseudomonas, BioSafe Beauveria, neem cake, yellow sticky traps, pheromone traps.

LANGUAGE: Reply in the same language the farmer writes in.
Telugu -> reply in Telugu | Hindi -> reply in Hindi | Tamil -> reply in Tamil | Default -> English

STRICT RULES:
1. Never suggest chemicals or synthetic inputs - organic only
2. If unsure, ask the farmer for more details
3. Always write cost in Rs (Indian Rupees)
4. Only give advice about chilli farming in Andhra Pradesh and Telangana
5. If the pest detection model gave results, use that as your starting point
"""

def call_groq(conversation, user_text, detection_context=None):
    full_text = user_text
    if detection_context:
        full_text = f"{detection_context}\n\nFarmer says: {user_text}"
    messages = (
        [{"role": "system", "content": SYSTEM_PROMPT}]
        + conversation
        + [{"role": "user", "content": full_text}]
    )
    response = client.chat.completions.create(
        model=MODEL, messages=messages,
        max_tokens=MAX_TOKENS, temperature=0.7,
    )
    return response.choices[0].message.content.strip()

def handle_image(image_path, description):
    print("\n   Checking your plant photo...", end="", flush=True)
    result = detector.detect(image_path)
    if result.get("success"):
        top = result["top_detection"]
        print(f" Looks like: {top['label']} ({top['confidence']}% sure)")
    else:
        print(f" Couldn't identify — will use your description instead")
    return detector.format_for_openai(result, description)

def banner():
    print()
    print("=" * 58)
    print("  ChilliGuru -- Your Chilli Farming Helper")
    print("  Andhra Pradesh & Telangana")
    print("=" * 58)
    print("  Ask me anything about your chilli crop!")
    print("  English / Telugu / Hindi / Tamil -- all welcome")
    print("-" * 58)
    print("  Type your question and press Enter")
    print("  To share a photo:  image /path/to/photo.jpg")
    print("  Examples:  help      To exit:  quit")
    print("=" * 58)
    print()

def help_text():
    print()
    print("  Try asking:")
    print("  * My chilli leaves are curling. What do I do?")
    print("  * Best time to plant Teja in Guntur?")
    print("  * White powder on my chilli leaves -- help!")
    print("  * naa mirapa aakulu pasapu ranguloki maarutunaai")
    print("  * meri mirch ke phool jhaD rahe hain")
    print()
    print("  To analyse a photo:")
    print("  * image /Users/yourname/Downloads/leaf.jpg")
    print()

def main():
    if GROQ_API_KEY == "YOUR_API_KEY_HERE":
        print("\n  Please set your Groq key:")
        print("  export GROQ_API_KEY='your-key-here'")
        print("  Get a free key at: https://console.groq.com\n")
        sys.exit(1)

    banner()
    conversation = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGood luck with your crop!\n")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q", "bye"):
            print("\nGood luck with your crop!\n")
            break
        if user_input.lower() == "help":
            help_text()
            continue

        detection_context = None

        if user_input.lower().startswith("image "):
            raw = user_input[6:].strip()
            img = Path(raw)
            if not img.exists():
                print(f"\n  Could not find photo at: {raw}")
                print("  Tip: drag the photo into the terminal to get the path\n")
                continue
            print(f"   Got your photo: {img.name}")
            desc       = input("   What problem are you seeing? (or press Enter to auto-check): ").strip()
            user_input = desc or "Please check this photo of my chilli plant and tell me what is wrong and how to fix it organically."
            detection_context = handle_image(str(img), user_input)
            if detection_context == "NOT_CHILLI":
                print("\nChilliGuru: Sorry! I can only analyse chilli or pepper plant photos.")
                print("            Please send a photo of your chilli plant and I will help you.\n")
                continue

        print("\nChilliGuru: ", end="", flush=True)
        try:
            reply = call_groq(conversation, user_input, detection_context)
            print(reply)
            conversation.append({"role": "user",      "content": user_input})
            conversation.append({"role": "assistant",  "content": reply})
        except Exception as e:
            print(f"  Something went wrong: {e}")
        print()

if __name__ == "__main__":
    main()
