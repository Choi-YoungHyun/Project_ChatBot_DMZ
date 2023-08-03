import os
import openai

openai.organization = "org-t59W5OwxkINFi0IhzgLvTQ8e"
openai.api_key = "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"

response = openai.Completion.create(
model="text-davinci-003",
prompt="Write a tagline for an ice cream shop."
)

print(response)
