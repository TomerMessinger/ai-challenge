# Gaddex · AI Engineer Technical Challenge

## Mini Proposal Builder

We are building the **Gaddex AI Proposal Builder**: a system that takes what a client asks for ("team building for 10 people in Málaga, something with go-karts") and puts together an event proposal using the services offered by our suppliers.

This challenge is a very small version of that system. We want to see how you work with a language model: how you get it to do exactly what you need and return reliable, structured output.

---

## ⏱️ Before you start

- **Suggested time: 3–4 hours.** Please don't spend much more than that. If you run out of time, explain what's missing in your README.
- **Simple and robust beats ambitious.** We are not looking for a complex architecture or a polished UI.
- **You may use AI coding assistants.** We are evaluating your judgment, not whether you type every line yourself.
- **Free choice of stack and model.** Python + SQLite is more than enough. Use any LLM provider you like (OpenAI, Anthropic, local models…). Do not commit your API keys.
- **If anything in the spec is unclear,** make a reasonable assumption and write it down in your README.

---

## 🎯 What you need to build

### Step 1 · Database with mock data

Create a simple database of **suppliers and their services** and fill it with mock data you generate yourself (by hand, with a script, with an LLM… your call).

- Between **30 and 50 services** spread across **Málaga, Madrid, Seville and Barcelona**.
- A variety of types: activities, restaurants/catering, transport, venues…
- Each service must have at least: supplier, city, type, description, price per person, and minimum and maximum capacity. Feel free to add more fields.
- All prices are **per person and VAT included** (don't overcomplicate this).

So that we can compare results across candidates, **your database must include these 5 services** (the rest is up to you):

| Supplier | City | Service | Price/person | Capacity | Notes |
|---|---|---|---|---|---|
| Karting Costa del Sol | Málaga | Go-kart Grand Prix (2h) | €55 | 8–20 | Outdoor |
| La Malagueña Catering | Málaga | Málaga-style tapas menu | €22 | 15–80 | Minimum 15 people. Vegetarian and vegan options |
| Restaurante El Patio | Madrid | Group dinner in private room | €48 | 10–60 | Vegan option |
| The Locked House | Madrid | Team escape room (1h30) | €24 | 12–30 | Not wheelchair accessible |
| Guadalquivir Kayak | Seville | River kayak tour (2h) | €35 | 6–40 | April to October only |

**Do not include suppliers in any city** other than the four listed above.

### Step 2 · From client request to a JSON proposal

Build a script (or a small API) that takes a **client request in free text** and returns a **proposal in JSON** using the services in your database.

The basic flow would look something like this:

1. The LLM interprets the request (city, number of people, budget, what they're in the mood for…).
2. You search the database for matching services.
3. The proposal is put together and returned as JSON.

How you split the work between the LLM and your own code is up to you, and it's one of the things we're most interested in.

**You design the JSON format.** Keep in mind that this JSON will be consumed by another system afterwards, for example a frontend or an LLM that lays out the proposal for the client. At a minimum, it should make it possible to know:

- What the system understood from the request.
- Which services are proposed (with their database ID), quantities and prices.
- The total and the price per person.
- Whether the client's request is met or not, plus any warnings or assumptions.
- What happens when the request can't be fulfilled or is missing information.

### Step 3 · Run it on the test briefs

[`data/briefs.json`](data/briefs.json) contains 6 client requests. Run your system on all of them and **save the outputs in `solution/outputs/`**. When interpreting dates, assume today is February 1, 2027.

Some are straightforward and others can't be fully met. We want to see how your system responds in both cases.

---

## 🚫 Three rules

1. **No made-up suppliers.** Everything in the proposal must exist in your database.
2. **The numbers must add up.** Prices, subtotals and totals must be correct and consistent with the database. How you guarantee this is up to you.
3. **The JSON must always be valid.** Even if the model fails, your system should never return garbage.

---

## ⭐ Optional extras

Only if you have time left. They are not needed to do a great job:

- Semantic search (embeddings) for preferences such as "something relaxed" or "outdoors".
- Returning 2 or 3 alternatives (budget, balanced, premium).
- A second step that turns the JSON into a formatted proposal (Markdown or HTML) using an LLM.
- A few simple automated tests.

---

## 📬 Submission

1. **Fork** this repository.
2. Build your solution inside the **`solution/`** folder of your fork. Do not modify this README or the files in `data/`.
3. Include in `solution/`:
   - Your code, including the script that creates and populates the database.
   - The `outputs/` folder with the output for each brief.
   - Your own **`README.md`** covering:
     - Requirements and how to run it (ideally in 2–3 commands), including how to configure the API key.
     - How you approached the use of the LLM (prompts, how you control the output, what the model does and what your code does).
     - Any assumptions you made.
     - What you would improve with more time.
     - Roughly how much time you spent.
4. Push your changes to your fork and send us the link.
