import telebot
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
import torch
from PIL import Image
import io

TOKEN = "7926616115:AAGyr9lgJ8kVPC47HJl8iEs6fIfOc0ULEE4"
bot = telebot.TeleBot(TOKEN)

text2img_pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to("cuda")
img2img_pipeline = StableDiffusionImg2ImgPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to("cuda")

@bot.message_handler(commands=['generate'])
def generate_images(message):
    prompt = message.text.replace('/generate', '').strip()
    if not prompt:
        bot.reply_to(message, "يرجى كتابة وصف للصورة بعد الأمر `/generate`.")
        return

    bot.reply_to(message, "⏳ جاري إنشاء الصور...")

    images = text2img_pipeline(
        prompt, 
        num_inference_steps=50, 
        guidance_scale=7.5, 
        num_images_per_prompt=4
    ).images

    for idx, img in enumerate(images):
        bio = io.BytesIO()
        img.save(bio, format='PNG')
        bio.seek(0)
        bot.send_photo(message.chat.id, bio)

print("✅ البوت يعمل الآن!")
bot.polling()
