import { NextResponse } from 'next/server';
import OpenAI from 'openai';
import Anthropic from '@anthropic-ai/sdk';

export const maxDuration = 300;

/* ============================================================================
   MODEL ROUTING LAYER — Step 1 of Material Studio rollout
   ----------------------------------------------------------------------------
   Each task is routed to the model best-suited for it:
     - Claude Opus 4.7   → Prize (Elite Lesson Plan) — ultimate deliverable only
     - Claude Sonnet 4.6 → Iterative init (activities/exceed) — verbatim extraction matters
     - Claude Haiku 4.5  → Main Analysis, IEP, iterative respond/reanalyze
     - gpt-4o-mini       → Chat, summaries, gap detector, gamifier
     - gpt-4o            → Legacy materializer (kept intact until Step 2 replaces it)
     - DALL-E 3          → Images (unchanged)

   A single `generateJSON()` helper routes by model prefix so prompts and
   call sites stay identical — only the model string changes per task.
   ============================================================================ */

// Model constants — edit one string to rebalance cost/quality for a single task
const MODEL_ELITE    = 'claude-opus-4-7';       // Prize only — the crown jewel
const MODEL_PREMIUM  = 'claude-sonnet-4-6';     // Iterative init (activity + exceed)
const MODEL_STANDARD = 'claude-haiku-4-5';      // Main analysis, IEP, iterative respond/reanalyze
const MODEL_LIGHT    = 'gpt-4o-mini';           // Chat, summaries, gap, gamifier

type ChatMessage = { role: 'system' | 'user' | 'assistant'; content: string };

// Unified JSON generation helper — dispatches to the right SDK based on model name
async function generateJSON(params: {
  model: string;
  maxTokens: number;
  systemPrompt: string;
  userContent: string;
  openai: OpenAI;
  anthropic: Anthropic;
}): Promise<string> {
  const { model, maxTokens, systemPrompt, userContent, openai, anthropic } = params;

  if (model.startsWith('claude-')) {
    // Anthropic: system goes in its own top-level field
    const resp = await anthropic.messages.create({
      model,
      max_tokens: maxTokens,
      system: systemPrompt,
      messages: [{ role: 'user', content: userContent }],
    });
    const text = resp.content
      .filter((b: any) => b.type === 'text')
      .map((b: any) => b.text)
      .join('');
    return text || '{}';
  } else {
    // OpenAI: system prompt as first message, JSON mode enabled
    const r = await openai.chat.completions.create({
      model,
      max_tokens: maxTokens,
      messages: [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: userContent },
      ],
      response_format: { type: 'json_object' },
    });
    return r.choices[0].message.content || '{}';
  }
}

// Chat helper for conversations with history. Currently only OpenAI is used
// for chat, but written provider-agnostically in case we route chat elsewhere.
async function generateChat(params: {
  model: string;
  maxTokens: number;
  systemPrompt: string;
  history: ChatMessage[];
  userMessage: string;
  openai: OpenAI;
  anthropic: Anthropic;
}): Promise<string> {
  const { model, maxTokens, systemPrompt, history, userMessage, openai, anthropic } = params;

  if (model.startsWith('claude-')) {
    const resp = await anthropic.messages.create({
      model,
      max_tokens: maxTokens,
      system: systemPrompt,
      messages: [
        ...history.map(m => ({ role: m.role as 'user' | 'assistant', content: m.content })),
        { role: 'user' as const, content: userMessage },
      ],
    });
    const text = resp.content
      .filter((b: any) => b.type === 'text')
      .map((b: any) => b.text)
      .join('');
    return text;
  } else {
    const r = await openai.chat.completions.create({
      model,
      max_tokens: maxTokens,
      messages: [
        { role: 'system', content: systemPrompt },
        ...history,
        { role: 'user', content: userMessage },
      ],
    });
    return r.choices[0].message.content || '';
  }
}

export async function POST(req: Request) {
  try {
    const body = await req.json();
    const { lessonText, config, selectedLenses, type, chatHistory, userMessage, lensContext } = body;

    // --- INPUT VALIDATION ---
    if (type && typeof type !== 'string') return NextResponse.json({ error: 'Invalid request type.' }, { status: 400 });
    if (!config) return NextResponse.json({ error: 'Missing config.' }, { status: 400 });

    const textRequiredTypes = ['prize','materializer','gamifier','iep','chat','iterative-init-activities','iterative-init-exceed','iterative-respond','iterative-reanalyze','iterative-summary','iterative-gap'];
    if (type && textRequiredTypes.includes(type) && (!lessonText || typeof lessonText !== 'string' || lessonText.trim().length === 0)) {
      return NextResponse.json({ error: 'Lesson text is required.' }, { status: 400 });
    }
    if (!type && (!lessonText || typeof lessonText !== 'string' || lessonText.trim().length === 0)) {
      return NextResponse.json({ error: 'Lesson text is required.' }, { status: 400 });
    }
    if (lessonText && lessonText.length > 50000) {
      return NextResponse.json({ error: 'Lesson text is too long. Please trim it to under 50,000 characters.' }, { status: 400 });
    }

    // --- API KEYS ---
    // Both keys required: OpenAI for light tasks, Anthropic for premium tasks.
    const openaiKey = process.env.OPENAI_API_KEY ?? '';
    const anthropicKey = process.env.ANTHROPIC_API_KEY ?? '';
    if (!openaiKey) return NextResponse.json({ error: 'OpenAI API key not configured.' }, { status: 401 });
    if (!anthropicKey) return NextResponse.json({ error: 'Anthropic API key not configured.' }, { status: 401 });

    const openai = new OpenAI({ apiKey: openaiKey });
    const anthropic = new Anthropic({ apiKey: anthropicKey });

    // --- PRIZE (Elite Lesson Plan) — CLAUDE OPUS 4.7 ---
    // Rare-fire feature (only on full mastery). Justifies the premium model.
    if (type === 'prize') {
      const prizePrompt = `You are an Elite Teacher Mentor. Adopt a "${config.tone}" tone throughout — this must shape your vocabulary, phrasing, and attitude in every section. Transform the following lesson into an elite-level lesson plan. Grade: ${config.grade}, Subject: ${config.subject}, Learner Profile: ${config.profile}, Time: ${config.minutes}m. Return ONLY a JSON object with EXACTLY these string keys: "Lesson Title", "Subject", "Grade Level", "Unit", "Section", "Objectives", "Materials Needed", "Anticipatory Set/Hook", "Direct Instruction", "Guided Practice", "Independent Practice", "Game Review", "Closure/Homework", "Assessment", "Differentiation". Every section must be written for ${config.profile} learners in a ${config.grade} ${config.subject} class. State allocated time at the start of each instructional phase. All phases must sum to exactly ${config.minutes}m.`;
      const content = await generateJSON({ model: MODEL_ELITE, maxTokens: 3500, systemPrompt: prizePrompt, userContent: lessonText, openai, anthropic });
      return NextResponse.json(JSON.parse(content || '{}'));
    }

    // --- MATERIALIZER (LEGACY — being replaced by Material Studio in Step 2) ---
    // Kept intact so existing mastery-unlock button still works during Step 1.
    // Will be deleted in Step 2 when Material Studio replaces it.
    // Uses gpt-4o because DALL-E-3 image pipeline is OpenAI-native and
    // HTML+image coordination is tightly coupled to the OpenAI response shape.
    if (type === 'materializer') {
      const matPrompt = `You are an Elite Teacher Mentor. Adopt a "${config.tone}" tone throughout. Create an EXTREMELY LONG, highly creative printable student Worksheet tailored specifically for ${config.profile} learners. Grade: ${config.grade}, Subject: ${config.subject}, Time: ${config.minutes}m.
Every activity, instruction, and item must be calibrated for ${config.grade} ${config.subject} ${config.profile} students within a ${config.minutes}-minute class.
TEACHER INSTRUCTIONS: "${userMessage || 'Create a comprehensive standard worksheet.'}" — STRICTLY FOLLOW THESE.
Return ONLY JSON: { "html": string, "requiresImage": boolean, "imagePrompt": string }.
HTML: fully styled inline CSS, readable fonts, generous spacing. Tables for grids. MINIMUM 7 items per activity. Put "{{IMAGE_PLACEHOLDER}}" where images go.`;
      const r = await openai.chat.completions.create({ model: 'gpt-4o', max_tokens: 6000, messages: [{ role: 'system', content: matPrompt }, { role: 'user', content: lessonText }], response_format: { type: 'json_object' } });
      let obj = JSON.parse(r.choices[0].message.content || '{}');
      if (!obj.html) obj.html = '';
      if (obj.requiresImage && obj.imagePrompt) {
        try {
          const fp = obj.imagePrompt + ' STRICT: 1 object. Black-and-white clipart line-art, white background. No humans/faces/eyes.';
          const [a, b, c] = await Promise.all([
            openai.images.generate({ model: 'dall-e-3', prompt: fp + ' (1)', n: 1, size: '1024x1024', response_format: 'b64_json' }),
            openai.images.generate({ model: 'dall-e-3', prompt: fp + ' (2)', n: 1, size: '1024x1024', response_format: 'b64_json' }),
            openai.images.generate({ model: 'dall-e-3', prompt: fp + ' (3)', n: 1, size: '1024x1024', response_format: 'b64_json' }),
          ]);
          const b64s = [a.data[0].b64_json || '', b.data[0].b64_json || '', c.data[0].b64_json || ''];
          if (b64s.every(x => x)) {
            const tag = `<div style="text-align:center;margin:30px 0;">${b64s.map(x => `<img src="data:image/png;base64,${x}" style="width:200px;height:200px;margin:15px;display:inline-block;border:2px dashed #000;padding:10px;" />`).join('')}</div>`;
            obj.html = obj.html.replace('{{IMAGE_PLACEHOLDER}}', tag);
          } else obj.html = obj.html.replace('{{IMAGE_PLACEHOLDER}}', '');
        } catch { obj.html = obj.html.replace('{{IMAGE_PLACEHOLDER}}', ''); }
      } else if (obj.html) obj.html = obj.html.replace('{{IMAGE_PLACEHOLDER}}', '');
      return NextResponse.json(obj);
    }

    // --- GAMIFIER (Kahoot CSV) — gpt-4o-mini ---
    // Simple structured CSV generation — cheapest model is fine.
    if (type === 'gamifier') {
      const gp = `You are an Elite Teacher Mentor. Adopt a "${config.tone}" tone. Create a 10-question MCQ trivia game perfectly calibrated for Grade ${config.grade} ${config.subject} ${config.profile} learners in a ${config.minutes}-minute class. Questions must match the vocabulary, complexity, and content expectations for ${config.grade} ${config.profile} students. Return ONLY JSON: { "csv": string }. CSV header: "Question,Answer 1,Answer 2,Answer 3,Answer 4,Time limit (sec),Correct answer(s)". Time limit 20. Correct answer 1-4. Output all 10 questions completely — do NOT truncate.`;
      const content = await generateJSON({ model: MODEL_LIGHT, maxTokens: 1500, systemPrompt: gp, userContent: lessonText, openai, anthropic });
      return NextResponse.json(JSON.parse(content || '{}'));
    }

    // --- IEP SCAFFOLD — CLAUDE HAIKU 4.5 ---
    // Sensitive pedagogical writing for accommodations. Haiku is a meaningful
    // upgrade over gpt-4o-mini here because the writing quality impacts a real
    // student's support plan.
    if (type === 'iep') {
      const ip = `You are an Elite Teacher Mentor. Adopt a "${config.tone}" tone. Create a custom micro-scaffold accommodation for this specific student: "${userMessage}". This scaffold is for use in a Grade ${config.grade} ${config.subject} class of ${config.profile} learners within a ${config.minutes}-minute period. The scaffold must account for both the individual student's needs AND the broader class context (${config.profile}). Return ONLY JSON: { "html": "fully styled HTML ready to print" }.`;
      const content = await generateJSON({ model: MODEL_STANDARD, maxTokens: 3000, systemPrompt: ip, userContent: lessonText, openai, anthropic });
      return NextResponse.json(JSON.parse(content || '{}'));
    }

    // --- CHAT (Ask the Mentor) — gpt-4o-mini ---
    // Conversational, fast response expected. gpt-4o-mini handles this well.
    if (type === 'chat') {
      if (!userMessage || typeof userMessage !== 'string' || !userMessage.trim()) {
        return NextResponse.json({ error: 'Chat message is required.' }, { status: 400 });
      }
      const sc = `You are a Mentor Coach in a TEXT CHAT with a teacher. Adopt a "${config.tone}" tone — this must shape how you phrase every sentence. Grade: ${config?.grade}, Subject: ${config?.subject}, Profile: ${config?.profile}, Time: ${config?.minutes}m. Lesson (500 chars): "${(lessonText || '').substring(0, 500)}". Focus: ${lensContext?.name}, Theory: ${lensContext?.theory}. Rules: warm, natural, concise. Use HTML with <br><br> spacing and inline CSS color headings. Reference their specific lesson. NO MARKDOWN. End with a question.`;

      // Validate history shape (from Fix #10 — still required)
      const validHistory: ChatMessage[] = (chatHistory || []).filter(
        (m: any) =>
          m &&
          typeof m === 'object' &&
          (m.role === 'user' || m.role === 'assistant') &&
          typeof m.content === 'string' &&
          m.content.trim().length > 0
      );

      const reply = await generateChat({
        model: MODEL_LIGHT,
        maxTokens: 800,
        systemPrompt: sc,
        history: validHistory,
        userMessage,
        openai,
        anthropic,
      });
      return NextResponse.json({ reply: reply.replace(/[*#]/g, '') });
    }

    // --- ITERATIVE SECTION 1: Activity & Section Feedback — CLAUDE SONNET 4.6 ---
    // Verbatim quote extraction is mission-critical (the lesson document only
    // updates correctly if the quote is findable by exact string search).
    // Sonnet is meaningfully more reliable at verbatim extraction than Haiku.
    if (type === 'iterative-init-activities') {
      const prompt = `You are an Elite Teacher Mentor — the absolute best in the world — conducting a rigorous, precise iterative review of a real teacher's lesson plan. Adopt a "${config.tone}" tone throughout every single field.

LESSON CONTEXT: Grade ${config.grade}, Subject: ${config.subject}, Learner Profile: ${config.profile}, Class Time: ${config.minutes} minutes.

YOUR TASK: Read the lesson text carefully. Identify between 4 and 6 distinct ACTIVITIES or LESSON SECTIONS that are EXPLICITLY present in this lesson (e.g., hook, warm-up, do now, direct instruction, modeling, guided practice, independent practice, group work, discussion, exit ticket, closure).

STRICT RULES — violating any of these is a failure:
1. ONLY include sections that genuinely exist in the lesson text with a real verbatim quote.
2. The "quote" field MUST be an EXACT verbatim substring copied character-for-character from the lesson text — max 25 words. It must be findable by exact string search. Do NOT paraphrase or summarize. Do NOT make up quotes.
3. "notFound" must ALWAYS be false — only include sections you can actually quote.
4. Spread feedback across the WHOLE lesson — beginning, middle, AND end. Do not cluster all feedback in one place.
5. Every field must be calibrated to Grade ${config.grade}, Subject ${config.subject}, ${config.profile} learners, ${config.minutes}-minute class.

For EACH activity/section return ALL of these fields:
- "id": unique string like "act_1", "act_2" etc.
- "sectionName": the name of this activity as it appears or can be clearly inferred from the lesson
- "quote": EXACT verbatim substring from the lesson — MAX 25 words — character-for-character copy
- "notFound": false (always)
- "feedback": THOROUGH, DEEPLY ANALYTICAL critique (minimum 4–6 sentences). You MUST: (1) Name the specific pedagogical weakness and explain WHY it is a weakness for ${config.profile} learners at Grade ${config.grade} in ${config.subject}. (2) Reference a specific educational researcher or theory by name. (3) Explain the concrete impact on student outcomes in a ${config.minutes}-minute class. (4) Identify exactly what is missing or underdeveloped. Be specific — no generic statements.
- "revision": A RICH, DETAILED, PEDAGOGICALLY ELEVATED rewrite of the quoted section. Must be immediately usable as a direct drop-in replacement for Grade ${config.grade} ${config.subject} ${config.profile} students in ${config.minutes} minutes. Length should match or slightly exceed the original quote.
- "priority": "HIGH" if the activity has a fundamental pedagogical flaw that directly harms learning outcomes. "MEDIUM" if it is a meaningful improvement opportunity.

Return ONLY valid JSON: { "feedbacks": [ { "id", "sectionName", "quote", "notFound", "feedback", "revision", "priority" } ] }`;

      const content = await generateJSON({ model: MODEL_PREMIUM, maxTokens: 7000, systemPrompt: prompt, userContent: lessonText, openai, anthropic });
      return NextResponse.json(JSON.parse(content || '{}'));
    }

    // --- ITERATIVE SECTION 2: Exceed Expectations Guide — CLAUDE SONNET 4.6 ---
    // Same verbatim-quote requirement as activities, same model.
    if (type === 'iterative-init-exceed') {
      const prompt = `You are an Elite Teacher Mentor — the absolute best in the world — building a precise, research-grounded "Exceed Expectations" guide for a real teacher's lesson. Adopt a "${config.tone}" tone throughout every field.

LESSON CONTEXT: Grade ${config.grade}, Subject: ${config.subject}, Learner Profile: ${config.profile}, Class Time: ${config.minutes} minutes.

Evaluate this lesson against EXACTLY these 5 pedagogical frameworks. The pioneer names are FIXED — use them exactly as written below, no substitutions:

1. Scaffolding — Pioneer: Lev Vygotsky — Theory: Zone of Proximal Development
2. Differentiation — Pioneer: Carol Ann Tomlinson — Theory: Differentiated Instruction
3. Culturally Responsive Teaching — Pioneer: Gloria Ladson-Billings — Theory: Culturally Relevant Pedagogy
4. Engagement — Pioneer: Phil Schlechty — Theory: Schlechty's Levels of Engagement
5. Objectives — Pioneer: Benjamin Bloom — Theory: Bloom's Taxonomy (Anderson & Krathwohl revision)

STRICT RULES — violating any of these is a failure:
1. Return EXACTLY 5 objects — one for each framework above. No more, no fewer.
2. The "category" field MUST use the exact category name from the list above.
3. The "pioneer" field MUST use the exact pioneer name from the list above — do not change these.
4. If hasSection is TRUE: the "quote" field MUST be an EXACT verbatim substring from the lesson text (max 25 words, character-for-character). Do NOT paraphrase. Do NOT invent quotes.
5. If hasSection is FALSE: "quote" must be an empty string "".
6. Every revision and addition must be calibrated to Grade ${config.grade}, Subject ${config.subject}, ${config.profile} learners, ${config.minutes} minutes.

For EACH framework return ALL of these fields:
- "category": exact framework name from the list above
- "pioneer": exact pioneer name from the list above — FIXED, do not change
- "hasSection": boolean — true if the lesson has meaningful content addressing this framework
- "quote": if hasSection TRUE → EXACT verbatim substring, max 25 words. If hasSection FALSE → ""
- "currentLevel": if hasSection TRUE → DETAILED honest assessment of current quality (3–4 sentences, specific to this lesson). If hasSection FALSE → ""
- "revision": RICH, THOROUGH improvement. If hasSection TRUE → rewrite the quoted section to TRULY EXCEED expectations with specific named strategies for Grade ${config.grade} ${config.subject} ${config.profile} in ${config.minutes} minutes (minimum 5–8 sentences). If hasSection FALSE → a complete ready-to-insert instructional section written in the same voice as the teacher's lesson (minimum 5–8 sentences).
- "addWhere": if hasSection FALSE → exact location in the lesson where this should be inserted (e.g., "After the warm-up activity, before direct instruction"). If hasSection TRUE → ""

Return ONLY valid JSON: { "guide": [ { "category", "pioneer", "hasSection", "quote", "currentLevel", "revision", "addWhere" } ] }`;

      const content = await generateJSON({ model: MODEL_PREMIUM, maxTokens: 7000, systemPrompt: prompt, userContent: lessonText, openai, anthropic });
      return NextResponse.json(JSON.parse(content || '{}'));
    }

    // --- ITERATIVE RESPOND — CLAUDE HAIKU 4.5 ---
    // Teacher pushed back or added context; AI updates a single card.
    // Smaller task than init generation, Haiku is right-sized.
    if (type === 'iterative-respond') {
      const { item, sectionType } = body;
      if (!item) return NextResponse.json({ error: 'Missing item.' }, { status: 400 });

      const safeStr = (v: any): string => String(v || '').replace(/`/g, "'").replace(/\$\{/g, '${');

      let prompt = '';
      if (sectionType === 'activity') {
        prompt = `You are an Elite Teacher Mentor. Adopt a "${config.tone}" tone. The teacher responded to your feedback on the "${safeStr(item.sectionName)}" section.
Original quote: "${safeStr(item.quote)}"
Teacher's response: "${userMessage}"
Grade: ${config.grade}, Subject: ${config.subject}, Profile: ${config.profile}, ${config.minutes}m.
Update your feedback and revision to reflect and directly address the teacher's response. "feedback" must be THOROUGH (4–6 sentences). "revision" must be RICH, DETAILED, immediately usable. Quote must be EXACT verbatim substring (max 25 words).
Return ONLY JSON: { "feedback": { "id": "${safeStr(item.id)}", "sectionName": "${safeStr(item.sectionName)}", "quote": "...", "feedback": "...", "revision": "...", "priority": "${safeStr(item.priority || 'MEDIUM')}", "notFound": false } }`;
      } else {
        prompt = `You are an Elite Teacher Mentor. Adopt a "${config.tone}" tone. The teacher responded to your "${safeStr(item.category)}" exceed-expectations guidance.
Teacher's response: "${userMessage}"
Grade: ${config.grade}, Subject: ${config.subject}, Profile: ${config.profile}, ${config.minutes}m.
Update your guidance. "currentLevel" must be DETAILED (3–4 sentences) if hasSection true. "revision" must be RICH and THOROUGH (5–8 sentences minimum). hasSection stays ${item.hasSection}. If true, quote must be EXACT substring max 25 words.
Return ONLY JSON: { "feedback": { "category": "${safeStr(item.category)}", "pioneer": "${safeStr(item.pioneer)}", "hasSection": ${item.hasSection}, "quote": "${safeStr(item.quote || '')}", "currentLevel": "...", "revision": "...", "addWhere": "${safeStr(item.addWhere || '')}" } }`;
      }
      const content = await generateJSON({ model: MODEL_STANDARD, maxTokens: 1800, systemPrompt: prompt, userContent: lessonText, openai, anthropic });
      return NextResponse.json(JSON.parse(content || '{}'));
    }

    // --- ITERATIVE REANALYZE — CLAUDE HAIKU 4.5 ---
    // Targeted single-card regeneration. Same reasoning as iterative-respond.
    if (type === 'iterative-reanalyze') {
      const { sectionType, sectionName, category } = body;
      let prompt = '';
      if (sectionType === 'activity') {
        prompt = `You are an Elite Teacher Mentor. Adopt a "${config.tone}" tone. Re-analyze the "${sectionName}" section in this updated lesson. Grade: ${config.grade}, Subject: ${config.subject}, Profile: ${config.profile}, ${config.minutes}m.
Find the best remaining improvement opportunity. "feedback" must be THOROUGH (4–6 sentences). "revision" must be RICH, DETAILED, immediately usable for Grade ${config.grade} ${config.subject} ${config.profile} in ${config.minutes} minutes. Quote must be EXACT verbatim substring max 25 words.
Return ONLY JSON: { "feedback": { "id": "act_r", "sectionName": "${sectionName}", "quote": "...", "feedback": "...", "revision": "...", "priority": "...", "notFound": false } }`;
      } else {
        prompt = `You are an Elite Teacher Mentor. Adopt a "${config.tone}" tone. Re-analyze the "${category}" framework in this updated lesson. Grade: ${config.grade}, Subject: ${config.subject}, Profile: ${config.profile}, ${config.minutes}m.
"currentLevel" must be a DETAILED honest assessment (3–4 sentences) if hasSection true. "revision" must be RICH and THOROUGH (5–8 sentences minimum) with specific named strategies for Grade ${config.grade} ${config.subject} ${config.profile} in ${config.minutes} minutes. If hasSection true, quote must be EXACT substring max 25 words.
Return ONLY JSON: { "feedback": { "category": "${category}", "pioneer": "...", "hasSection": ..., "quote": "...", "currentLevel": "...", "revision": "...", "addWhere": "..." } }`;
      }
      const content = await generateJSON({ model: MODEL_STANDARD, maxTokens: 1800, systemPrompt: prompt, userContent: lessonText, openai, anthropic });
      return NextResponse.json(JSON.parse(content || '{}'));
    }

    // --- ITERATIVE SUMMARY — gpt-4o-mini ---
    // Short encouraging text, cheapest model is fine.
    if (type === 'iterative-summary') {
      const { changelog } = body;
      const sp = `You are an Elite Teacher Mentor. Adopt a "${config.tone}" tone. The teacher made these improvements to their Grade ${config.grade} ${config.subject} ${config.profile} lesson (${config.minutes}-minute class):
${(changelog || []).map((c: any, i: number) => `${i + 1}. [${c.sectionName}] "${c.isAddition ? '(new addition)' : c.quote}" → "${c.revision}"`).join('\n')}
Write a warm, encouraging 3–4 sentence summary in a "${config.tone}" voice explaining what improved and why it strengthens the lesson for ${config.grade} ${config.profile} students in ${config.minutes} minutes. Be specific. End with one concrete next step.
Return ONLY JSON: { "summary": "..." }`;
      const content = await generateJSON({ model: MODEL_LIGHT, maxTokens: 600, systemPrompt: sp, userContent: lessonText, openai, anthropic });
      return NextResponse.json(JSON.parse(content || '{}'));
    }

    // --- ITERATIVE GAP DETECTOR — gpt-4o-mini ---
    // Simple yes/no check per framework. Cheapest model is fine.
    if (type === 'iterative-gap') {
      const gp = `You are an Elite Teacher Mentor. Adopt a "${config.tone}" tone. Review this revised lesson for Grade ${config.grade}, Subject: ${config.subject}, Profile: ${config.profile}, ${config.minutes} minutes. Check if it NOW adequately addresses: 1. Scaffolding, 2. Differentiation, 3. Culturally Responsive Teaching, 4. Engagement, 5. Objectives.
For each: "category", "adequatelyAddressed" (boolean), "note" (if not adequately addressed — one concrete sentence on what is still missing, written for Grade ${config.grade} ${config.subject} ${config.profile} in ${config.minutes} minutes).
Return ONLY JSON: { "gaps": [ { "category", "adequatelyAddressed", "note" } ] }`;
      const content = await generateJSON({ model: MODEL_LIGHT, maxTokens: 600, systemPrompt: gp, userContent: lessonText, openai, anthropic });
      return NextResponse.json(JSON.parse(content || '{}'));
    }

    // --- MAIN ANALYSIS (Full / Focused / Custom) — CLAUDE HAIKU 4.5 ---
    // Genuine upgrade from gpt-4o-mini at low extra cost. Runs on every session
    // (often multiple times for Full Report split into chunks), so cost matters —
    // Haiku strikes the right quality/cost balance for this high-volume task.
    const knownTypes = ['prize','materializer','gamifier','iep','chat','iterative-init-activities','iterative-init-exceed','iterative-respond','iterative-reanalyze','iterative-summary','iterative-gap'];
    if (type && !knownTypes.includes(type)) {
      return NextResponse.json({ error: `Unknown request type: ${type}` }, { status: 400 });
    }

    if (config.mode === 'Custom selection' && (!selectedLenses || selectedLenses.length === 0)) {
      return NextResponse.json({ error: 'No categories selected for custom mode.' }, { status: 400 });
    }

    const PIONEER_INSTRUCTIONS = `MANDATORY — Pioneer names are FIXED. Use EXACTLY these, no changes allowed:
Clarity → John Hattie | Alignment → Ralph Tyler | Inclusivity → David Rose & Anne Meyer
Scaffolding → Lev Vygotsky | Differentiation → Carol Ann Tomlinson | Objectives → Benjamin Bloom
Assessments → Dylan Wiliam | Engagement → Phil Schlechty | Strategies → Robert Marzano
Materials → Grant Wiggins | Collaboration → David & Roger Johnson | Closure → Madeline Hunter`;

    let reportCommand = '';
    if (config.mode === 'Focused report') {
      reportCommand = `FOCUSED REPORT MODE.
You MUST return EXACTLY 3 feedback objects — no more, no fewer. This is a hard requirement.
Choose the 3 categories where THIS specific lesson has the GREATEST room for improvement.
Do NOT return 4, 5, 6, or any other number. Returning anything other than exactly 3 objects is a failure.
The 3 categories must come from: Clarity, Alignment, Inclusivity, Scaffolding, Differentiation, Objectives, Assessments, Engagement, Strategies, Materials, Collaboration, Closure.`;
    } else if (config.mode === 'Custom selection') {
      const catList = selectedLenses.join(', ');
      reportCommand = `CUSTOM SELECTION MODE.
Return EXACTLY ${selectedLenses.length} feedback object(s) — one for each of these categories and NO others: ${catList}.
Do NOT include any category not in this list. Do NOT add extra categories. Do NOT substitute different categories.
Allowed categories (ONLY these): ${catList}.
Return EXACTLY ${selectedLenses.length} object(s), no more, no fewer.`;
    } else {
      reportCommand = `FULL REPORT MODE.
Return EXACTLY 12 objects — one for EACH of ALL 12 categories: Clarity, Alignment, Inclusivity, Scaffolding, Differentiation, Objectives, Assessments, Engagement, Strategies, Materials, Collaboration, Closure.
Do not omit any category. Do not add extras.`;
    }

    const systemPrompt = `You are an Elite Teacher Mentor. Analyze this lesson for ${config.grade} ${config.subject} (${config.profile} learners). Tone: "${config.tone}". Class duration: ${config.minutes} minutes.

${reportCommand}

${PIONEER_INSTRUCTIONS}

For EACH category object return ALL of these fields:
- "id": unique string
- "name": exact category name from the allowed list
- "pioneer": EXACTLY as specified in the pioneer instructions above — do not change these names under any circumstances
- "theory": 90 words explaining the research theory behind this category
- "lessonFeedback": 100 words of specific feedback on THIS lesson through this lens — reference actual content from the lesson
- "upgrade": 100 words of concrete, actionable suggestions to improve this specific lesson
- "example": 150 words describing a precise instructional routine for Grade ${config.grade} ${config.subject} ${config.profile} students in a ${config.minutes}-minute class
- "quiz": exactly 5 multiple-choice questions, each with "question" (string), "options" (array of exactly 4 strings), "correct" (one of the 4 option strings exactly as written)

Return JSON: { "feedback": [ { "id", "name", "pioneer", "theory", "lessonFeedback", "upgrade", "example", "quiz" } ] }`;

    const content = await generateJSON({ model: MODEL_STANDARD, maxTokens: 16000, systemPrompt, userContent: lessonText, openai, anthropic });
    return NextResponse.json(JSON.parse(content || '{}'));

  } catch (error: any) {
    console.error('[API Error]', error);
    const isKnown = error?.message?.includes('JSON') || error?.message?.includes('token') || error?.message?.includes('rate limit');
    const safeMessage = isKnown
      ? 'Generation failed. Try again or use a shorter lesson.'
      : 'Something went wrong. Please try again.';
    return NextResponse.json({ error: safeMessage }, { status: 500 });
  }
}
