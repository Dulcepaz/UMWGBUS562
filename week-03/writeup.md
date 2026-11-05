## What I Built

For this assignment, I built a RAG tool using Claude Code. The tool searches documents and uses the information it finds to answer questions. 
I started with Copperline documents and then changed the tool to work with the Larkfield Credit Union documents. I also added a way to save the questions, answers and sources in text files. 
I tested six quetions and saved seven required runs becaues I had to run Question 1 before and after fixing the outdated document. 
What suprised me was that I found both the old $35 fee and the new $29 fee. I learned that it is very important to double-check the information because AI can still make mistakes. 

## Why This Approch 
Chunking method: structure
makes sense on its own: Partly. Most chunks make sense by themselves, but some need information from other sections to fully understand the policy. 
Label and source: Both. Each chuk includes the doucment tittle ands section heading. The tool also saves the source filename so we know where the information cam from.
Short enough: Yes. The Larkfield chunks are between 30 and 116 words, which makes them short enough for the embedding model to read. 

## Example Run

[1] similarity 0.638 | source: consumer-fee-schedule.html
Consumer Fee Schedule › Wire transfers

- Incoming domestic wire transfer: $10.
- Incoming international wire transfer: $15.
- Outgoing domestic wire transfer: $25.
- Outgoing international wire transfer: $45. The receiving bank or an intermediary bank may deduct its own fees from the amount sent.

[2] similarity 0.610 | source: fee-refund-guide.html
Fee Refund Guide for Member Service Staff › Refunds that need a branch manager

A refund needs the branch manager's approval when the fee is more than $30, when the member has already had a refund in the last 90 days, or when the member asks for several fees to be refunded at once. A branch manager may approve refunds of up to $150 in total for one member in any 90-day period.

[3] similarity 0.573 | source: fee-refund-guide.html
Fee Refund Guide for Member Service Staff › What a member service representative can approve

A member service representative may approve a refund of a single fee of $30 or less without anyone else's approval. A representative may approve only one refund for the same member in any 90-day period.

[4] similarity 0.523 | source: fee-refund-guide.html
Fee Refund Guide for Member Service Staff › Refunds that need the Member Services Director

Refunds of more than $150 in total for one member in any 90-day period need the Member Services Director's approval. So does any refund for a business account.

ANSWER
No. The outgoing international wire transfer fee is $45 [consumer-fee-schedule.html]. A member service representative can only approve a refund without further approval if the single fee is $30 or less [fee-refund-guide.html]. Since $45 exceeds that threshold, it falls under refunds that need the branch manager's approval, which applies whenever a fee is more than $30 [fee-refund-guide.html].

So this refund would require branch manager approval (and the branch manager can approve up to $150 total for that member in a 90-day period) [fee-refund-guide.html].

## Reflection

What was wrong: The AI found both the old $35 overdraft fee and the new $29 fee. This showed me taht AI can find outdated information if old documents are still in the system. 

How I fixed it: I moved the old document into a separate archive folder and rebuilt the database. this helped make sure the tool only used updated information. 

Prevention: Archive process. I think the business should have a process to remove old documents whenever a new policie is added. This would help prevent wrong answers. 


Question 6: Decline. The AI did not give a mortgage rate because the document did not contain that information. This showed me the model was working correctly becasue it did not guess an answer. 


