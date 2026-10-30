## What I Built 
I created a Prompt Lab using Claude’s API. I completed four tasks for Wildframe Coffee Roasters. 

## Why This Approach 

Task 1 Technique: Zero-shot prompting: This technique worked well because claude created six subscriber perks without needing examples. The results followed the instructions. 


Task 2 Technique: One-shot prompting: Giving Claude one example helped it follow the same format for all three coffee cards. The results were clear and consistent. 


Task 3 Technique: Few-shot: The three examples helped Claude understand the company’s tone. However, Claude made an assumption about selling decaf, so I had to ask it to correct the response. 
 

Task 4 Technique: Chain-of-thought prompting: This technique helped with the math, compared all three locations, and recommended the best option based on the numbers. 

## Example Run:
I selected Task 4 technique of chain-of-thought because it shows how Claude used calculations to compare three locations and recommend the best option. 


claude-sonnet-5

### You

Wildframe Coffee Roaster wants to open a pop-up coffee stand for four weeks this spring. 

The total rent budget is $5,000 for all four weeks.

Compare these three locations using the data below. 

Location 1: Riverside Farmers Market
- 9,000 visitors per week 
- 2% of visitors buy
- Average sale: $14
- Weekly rent: $900

Location 2: Union Station Concourse 
- 30,000 visitors per week
- 0.5% of visitors buy 
- Average sale: $6
- Weekly rent: $2,000

Location 3: Campus Library atrium
- 6,000 visitors per week 
- 3% of visitors buy 
- Average sale: $5
- Weekly rent: $600

Please work through the calculations step by step.

For EACH of the THREE locations:

1. Calculate expected weekly customers.
2. Calculate expected weekly sales revenue.
3. Calculate total rent for four weeks.
4. Calculate expected revenue for four weeks.
5. Subtract four-week rent from four-week revenue. 

Compare all three locations.

Explain which locations meet the $5,000 total rent budget. 

Recommend the best location based on the numbers.

Use simple english and show your calculations clearly.

Required format:
End your response with a single line starting with "Recommendation:" followed by only the name of the best location.

### Claude

# Wildframe Coffee Roaster – Pop-Up Location Analysis

## Location 1: Riverside Farmers Market

**1. Weekly customers**
9,000 visitors × 2% = **180 customers/week**

**2. Weekly revenue**
180 customers × $14 = **$2,520/week**

**3. Total rent (4 weeks)**
$900 × 4 = **$3,600**

**4. Total revenue (4 weeks)**
$2,520 × 4 = **$10,080**

**5. Profit (Revenue − Rent)**
$10,080 − $3,600 = **$6,480**

---

## Location 2: Union Station Concourse

**1. Weekly customers**
30,000 visitors × 0.5% = **150 customers/week**

**2. Weekly revenue**
150 customers × $6 = **$900/week**

**3. Total rent (4 weeks)**
$2,000 × 4 = **$8,000**

**4. Total revenue (4 weeks)**
$900 × 4 = **$3,600**

**5. Profit (Revenue − Rent)**
$3,600 − $8,000 = **-$4,400** (a loss)

---

## Location 3: Campus Library Atrium

**1. Weekly customers**
6,000 visitors × 3% = **180 customers/week**

**2. Weekly revenue**
180 customers × $5 = **$900/week**

**3. Total rent (4 weeks)**
$600 × 4 = **$2,400**

**4. Total revenue (4 weeks)**
$900 × 4 = **$3,600**

**5. Profit (Revenue − Rent)**
$3,600 − $2,400 = **$1,200**

---

## Comparison Table

| Location | 4-Week Rent | 4-Week Revenue | Profit | Fits $5,000 Budget? |
|---|---|---|---|---|
| Riverside Farmers Market | $3,600 | $10,080 | **$6,480** | ✅ Yes |
| Union Station Concourse | $8,000 | $3,600 | **-$4,400** | ❌ No |
| Campus Library Atrium | $2,400 | $3,600 | **$1,200** | ✅ Yes |

---

## Budget Check

- **Riverside Farmers Market**: $3,600 rent — fits easily within the $5,000 budget, leaving $1,400 unused.
- **Union Station Concourse**: $8,000 rent — **over budget by $3,000**, so this location is not even affordable.
- **Campus Library Atrium**: $2,400 rent — fits well within budget, leaving $2,600 unused.

---

## Recommendation

Union Station Concourse is both the most expensive option and the worst performer — it goes over budget and actually loses money. Campus Library Atrium is affordable and profitable but earns much less than Riverside.

Riverside Farmers Market is the clear winner: it fits comfortably within the budget and produces by far the highest profit ($6,480), more than 5 times the profit of Campus Library Atrium.

Recommendation: Riverside Farmers Market


## Reflection:

In this assignment, I learned that each prompting technique works differently. Zero-shot was easy because Claude did not need examples to give me six ideas. One-shot worked well because the coffee cards all had the same format. Few-shot helped Claude understand how to reply to customers, but it still made a mistake about the decaf coffee. I had to ask Claude to fix it. For the last task, chain-of-thought helped me see the math and understand why one location was better than the others to make a more informed decision. I think all four techniques worked well, but some needed more instructions than others. I also learned that even when I give AI clear instructions, I still need to check the answers because they are not always correct. I think that chain-of-thought made the biggest difference because it helped me organize and understand the calculations and choose the best location. I am least sure about few-shot because Claude made a mistake about the decaf coffee. Next time, I think I would try giving clearer instructions so Claude does not assume information. 
