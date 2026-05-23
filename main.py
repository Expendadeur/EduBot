import os
import json
import asyncio
import time
from typing import TypedDict, List, Optional, AsyncIterator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langgraph.graph import StateGraph, END
import groq as groq_lib
import google.generativeai as genai

# ── Configuration ────────────────────────────────────────────
GROQ_API_KEY   = os.getenv("GROQ_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

groq_client  = groq_lib.Groq(api_key=GROQ_API_KEY)
genai.configure(api_key=GEMINI_API_KEY)

gemini_flash   = genai.GenerativeModel("gemini-1.5-flash")
gemini_pro     = genai.GenerativeModel("gemini-1.5-pro")
gemini_flash2  = genai.GenerativeModel("gemini-2.0-flash-exp")

AUTHOR_SIGNATURE = """

---
Reponse generee par EduBot - Assistant Pedagogique Intelligent
Janvier NZAMBIMANA | Etudiant en M1, ITN
Besoin d'aide ou signaler une erreur ? Contactez:janviernzambimana91@gmail.com

Ce chatbot est gratuit et disponible pour tous. Si vous le trouvez utile,
vous pouvez soutenir son developpement en faisant un don via :
Mobile Money / Lumicash : +257 68 58 97 29 (Janvier NZAMBIMANA)
Chaque contribution, meme petite, aide a maintenir et ameliorer EduBot. Merci !

IMPORTANT : Veuillez verifier les informations ci-dessus avant de les utiliser ou de les confirmer.
Les reponses de l'IA peuvent contenir des inexactitudes."""

BASE_PROMPT = """Tu es EduBot, assistant pedagogique professionnel cree par Janvier NZAMBIMANA, etudiant en M1 ITN.
Tu aides les etudiants africains et du monde entier dans leurs apprentissages.
Reponds toujours en francais sauf si l'utilisateur ecrit dans une autre langue, alors reponds dans cette langue.
Sois clair, pedagogique, bienveillant, precis et encourage toujours l'etudiant.
Structure tes reponses avec des titres et sections claires quand c'est utile.
Evite les emojis dans tes reponses.
A la fin de chaque reponse, rappelle a l'utilisateur de verifier les informations avant de les confirmer."""

# ── Etat LangGraph ───────────────────────────────────────────
class AgentState(TypedDict):
    question:    str
    history:     List[dict]
    topic:       str
    response:    str
    model_used:  str
    quality_ok:  bool
    retry_count: int
    app_type:    str
    code_plan:   List[str]
    code_parts:  List[str]
    language:    str

# ── Detection de langue ──────────────────────────────────────
def detect_language(text: str) -> str:
    english_words = ["the","is","are","what","how","why","when","where","who","i","you","we","please","help","can"]
    swahili_words = ["ni","na","ya","kwa","katika","nini","jinsi","tafadhali","msaada"]
    kirundi_words = ["ndi","ni","kuri","muri","ivyo","gute","ingene","murakoze","ndabashimiye"]
    
    text_lower = text.lower()
    words = text_lower.split()
    
    en_score = sum(1 for w in words if w in english_words)
    sw_score = sum(1 for w in words if w in swahili_words)
    ki_score = sum(1 for w in words if w in kirundi_words)
    
    if en_score > sw_score and en_score > ki_score and en_score >= 2:
        return "en"
    elif sw_score > ki_score and sw_score >= 2:
        return "sw"
    elif ki_score >= 2:
        return "rn"
    return "fr"

def get_language_prompt(lang: str) -> str:
    prompts = {
        "en": BASE_PROMPT + "\nThe user writes in English, respond in English.",
        "sw": BASE_PROMPT + "\nMtumiaji anaandika kwa Kiswahili, jibu kwa Kiswahili.",
        "rn": BASE_PROMPT + "\nUmukoresha andika mu Kirundi, subiza mu Kirundi.",
        "fr": BASE_PROMPT,
    }
    return prompts.get(lang, BASE_PROMPT)

# ── Superviseur ──────────────────────────────────────────────
def supervisor_node(state: AgentState) -> AgentState:
    q    = state["question"].lower()
    lang = detect_language(state["question"])

    app_triggers = [
        "cree une app","creer une application","code une app",
        "developpe une app","fais-moi une app","make an app",
        "create an app","build an app","cree un programme complet",
        "cree un site","cree un systeme","code complet","create a game",
        "application complete","projet complet","cree un jeu","fais un jeu",
        "fais une application","developpe un systeme","construis une app",
    ]
    is_app_request = any(t in q for t in app_triggers)

    flutter_kw = ["flutter","dart","mobile","application mobile"]
    web_kw     = ["html","css","javascript","react","vue","angular","site web","frontend","backend"]
    sql_kw     = ["sql","base de donnees","mysql","postgresql","sqlite","mongodb","database"]
    python_kw  = ["python","django","flask","fastapi","script python"]
    tech_kw    = [
        "code","programme","algorithme","fonction","boucle","classe","api","bug",
        "erreur","reseau","linux","git","docker","kubernetes","serveur","cybersecurite",
        "machine learning","intelligence artificielle","data science","ia","ml",
    ]
    science_kw = [
        "mathematiques","physique","chimie","equation","integrale","derivee",
        "vecteur","biologie","atome","formule","statistiques","probabilites",
        "thermodynamique","mecanique","optique","electricite",
    ]
    human_kw   = [
        "histoire","philosophie","economie","politique","litterature","psychologie",
        "sociologie","droit","colonialisme","revolution","geographie","culture",
        "religion","ethique","anthropologie",
    ]
    health_kw  = [
        "sante","maladie","medecine","symptome","traitement","nutrition","sport",
        "bien-etre","mental","depression","anxiete","hopital","medicament",
    ]

    if is_app_request:
        app_type = "python"
        if any(k in q for k in flutter_kw):  app_type = "flutter"
        elif any(k in q for k in web_kw):    app_type = "web"
        elif any(k in q for k in sql_kw):    app_type = "sql"
        elif any(k in q for k in python_kw): app_type = "python"
        return {**state, "topic": "coder", "app_type": app_type, "language": lang}

    s_tech    = sum(1 for k in tech_kw    if k in q)
    s_science = sum(1 for k in science_kw if k in q)
    s_human   = sum(1 for k in human_kw   if k in q)
    s_health  = sum(1 for k in health_kw  if k in q)

    scores = {"tech": s_tech, "science": s_science, "human": s_human, "health": s_health}
    best   = max(scores, key=scores.get)
    topic  = best if scores[best] > 0 else "general"

    return {**state, "topic": topic, "app_type": "", "language": lang}

# ── Agent Codeur ─────────────────────────────────────────────
def agent_coder_node(state: AgentState) -> AgentState:
    q        = state["question"]
    app_type = state.get("app_type", "python")
    lang     = state.get("language", "fr")
    sys_prompt = get_language_prompt(lang)

    plan_prompt = f"""Tu es expert developpeur senior. L'etudiant demande: "{q}"
Genere un plan JSON UNIQUEMENT (aucun texte avant ou apres, pas de balises markdown) :
{{"app_name":"...","description":"...","technology":"{app_type}","etapes":["Etape 1: ...","Etape 2: ...","Etape 3: ...","Etape 4: ..."],"fichiers":["fichier1","fichier2"],"dependances":["dep1","dep2"]}}"""

    code_plan = ["Analyse des besoins", "Conception de l'architecture", "Developpement", "Tests et validation"]
    try:
        plan_resp = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": plan_prompt}],
            max_tokens=700, temperature=0.2,
        )
        raw   = plan_resp.choices[0].message.content.strip()
        raw   = raw.replace("```json", "").replace("```", "").strip()
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        if start != -1 and end > start:
            plan_data = json.loads(raw[start:end])
            code_plan = plan_data.get("etapes", code_plan)
    except Exception as e:
        print(f"Erreur plan JSON (ignoree): {e}")

    code_prompt = f"""{sys_prompt}

L'etudiant demande : "{q}"
Technologie principale : {app_type}

Genere le CODE COMPLET professionnel avec la structure suivante :

## Description du projet
[description claire en 2-3 phrases]

## Plan de developpement
[etapes numerotees]

## Prerequis et installation
[commandes d'installation des dependances]

## Code complet

### Fichier : [nom_du_fichier.extension]
```{app_type if app_type not in ['flutter','web'] else ('dart' if app_type=='flutter' else 'html')}
[code complet, fonctionnel, bien commente, sans emojis dans les commentaires]
```

[repeter pour chaque fichier si necessaire]

## Comment executer le projet
[instructions pas-a-pas, precises]

## Structure du projet
[arborescence des fichiers]

## Tests recommandes
[comment tester que tout fonctionne]

## Ameliorations possibles
[3 a 5 idees d'amelioration]

Le code doit etre 100% fonctionnel, bien structure, avec des commentaires pedagogiques clairs.
N'utilise PAS d'emojis dans le code ou les commentaires."""

    full_code  = ""
    model_used = ""

    models_to_try = [
        ("groq_llama3_70b",   lambda: _groq_call_model("llama-3.3-70b-versatile", [{"role":"system","content":sys_prompt},{"role":"user","content":code_prompt}], 4096)),
        ("gemini_flash2",     lambda: _gemini_model_call(gemini_flash2, code_prompt)),
        ("gemini_pro",        lambda: _gemini_model_call(gemini_pro,   code_prompt)),
        ("groq_mixtral",      lambda: _groq_call_model("mixtral-8x7b-32768", [{"role":"system","content":sys_prompt},{"role":"user","content":code_prompt}], 4096)),
        ("gemini_flash",      lambda: _gemini_model_call(gemini_flash, code_prompt)),
    ]

    for name, fn in models_to_try:
        try:
            full_code  = fn()
            model_used = name
            break
        except Exception as e:
            print(f"Modele {name} echoue: {e}")
            continue

    if not full_code:
        full_code  = "Impossible de generer le code pour le moment. Veuillez reessayer."
        model_used = "Erreur - tous les modeles indisponibles"

    return {**state, "response": full_code + AUTHOR_SIGNATURE,
            "model_used": model_used, "code_plan": code_plan}

# ── Helpers modeles ───────────────────────────────────────────
def _groq_call_model(model: str, messages: list, max_tokens: int = 2048) -> str:
    r = groq_client.chat.completions.create(
        model=model, messages=messages,
        max_tokens=max_tokens, temperature=0.6,
    )
    return r.choices[0].message.content

def _gemini_model_call(model, prompt: str) -> str:
    return model.generate_content(prompt).text

def _groq_call(messages: list, max_tokens: int = 2048) -> str:
    return _groq_call_model("llama-3.3-70b-versatile", messages, max_tokens)

def _msgs(state: AgentState, system: str) -> list:
    m = [{"role": "system", "content": system}]
    for h in state["history"][-8:]:
        m.append({"role": h["role"], "content": h["content"]})
    m.append({"role": "user", "content": state["question"]})
    return m

def _try_models_for_response(state: AgentState, sys_prompt: str, extra_prompt: str = "") -> tuple:
    question = state["question"]
    prompt_text = f"{sys_prompt}\n\n{extra_prompt}\n\nQuestion : {question}\n\nReponse :"

    attempts = [
        ("groq_llama3_70b",  lambda: _groq_call(_msgs(state, sys_prompt))),
        ("gemini_flash2",    lambda: _gemini_model_call(gemini_flash2, prompt_text)),
        ("gemini_flash",     lambda: _gemini_model_call(gemini_flash,  prompt_text)),
        ("gemini_pro",       lambda: _gemini_model_call(gemini_pro,    prompt_text)),
        ("groq_mixtral",     lambda: _groq_call_model("mixtral-8x7b-32768", _msgs(state, sys_prompt))),
    ]
    for name, fn in attempts:
        try:
            result = fn()
            return result, name
        except Exception as e:
            print(f"Modele {name} echoue: {e}")
    return "Erreur : aucun modele disponible. Veuillez reessayer.", "Erreur"

# ── Agents specialises ────────────────────────────────────────
def agent_tech_node(state: AgentState) -> AgentState:
    lang       = state.get("language", "fr")
    sys_prompt = get_language_prompt(lang)
    extra      = """Tu es expert en informatique et technologies.
Reponds de facon structuree avec des exemples concrets quand pertinent.
N'utilise pas d'emojis."""
    r, model = _try_models_for_response(state, sys_prompt, extra)
    return {**state, "response": r + AUTHOR_SIGNATURE, "model_used": f"{model} (Tech)"}

def agent_human_node(state: AgentState) -> AgentState:
    lang       = state.get("language", "fr")
    sys_prompt = get_language_prompt(lang)
    extra      = """Tu es expert en sciences humaines, histoire, philosophie, economie et droit.
Fournis des analyses nuancees avec des references historiques et contextuelles.
N'utilise pas d'emojis."""
    hist = "".join(
        f"{'Etudiant' if m['role']=='user' else 'EduBot'}: {m['content']}\n"
        for m in state["history"][-8:]
    )
    prompt_text = f"{sys_prompt}\n\n{extra}\n\nHistorique:\n{hist}\nEtudiant: {state['question']}\nEduBot:"
    try:
        r     = _gemini_model_call(gemini_flash2, prompt_text)
        model = "gemini_flash2"
    except:
        try:
            r     = _gemini_model_call(gemini_pro, prompt_text)
            model = "gemini_pro"
        except:
            r, model = _try_models_for_response(state, sys_prompt, extra)
    return {**state, "response": r + AUTHOR_SIGNATURE, "model_used": f"{model} (Humanites)"}

def agent_science_node(state: AgentState) -> AgentState:
    lang       = state.get("language", "fr")
    sys_prompt = get_language_prompt(lang)
    extra      = """Tu es expert en sciences exactes : mathematiques, physique, chimie, biologie.
Explique les concepts avec rigueur, montre les formules et calculs etape par etape.
N'utilise pas d'emojis."""
    r, model = _try_models_for_response(state, sys_prompt, extra)
    return {**state, "response": r + AUTHOR_SIGNATURE, "model_used": f"{model} (Sciences)"}

def agent_health_node(state: AgentState) -> AgentState:
    lang       = state.get("language", "fr")
    sys_prompt = get_language_prompt(lang)
    extra      = """Tu es assistant sante educatif. Fournis des informations generales sur la sante et le bien-etre.
IMPORTANT : rappelle toujours que pour tout probleme de sante, l'utilisateur doit consulter un professionnel de sante.
N'utilise pas d'emojis."""
    r, model = _try_models_for_response(state, sys_prompt, extra)
    return {**state, "response": r + AUTHOR_SIGNATURE, "model_used": f"{model} (Sante)"}

def agent_general_node(state: AgentState) -> AgentState:
    lang       = state.get("language", "fr")
    sys_prompt = get_language_prompt(lang)
    extra      = "Reponds de maniere claire, complete et pedagogique. N'utilise pas d'emojis."
    r, model   = _try_models_for_response(state, sys_prompt, extra)
    return {**state, "response": r + AUTHOR_SIGNATURE, "model_used": f"{model} (General)"}

# ── Verificateur de qualite ───────────────────────────────────
def quality_checker_node(state: AgentState) -> AgentState:
    r     = state.get("response", "")
    retry = state.get("retry_count", 0)
    errors = ["erreur : aucun modele", "impossible de generer", "error"]
    is_error   = any(e in r.lower()[:60] for e in errors)
    is_too_short = len(r.strip()) < 100
    ok = not is_error and not is_too_short
    if not ok and retry < 2:
        return {**state, "quality_ok": False, "retry_count": retry + 1}
    return {**state, "quality_ok": True}

# ── Routeurs ──────────────────────────────────────────────────
def route_topic(state: AgentState):
    return state["topic"]

def route_quality(state: AgentState):
    return "end" if state["quality_ok"] else "retry"

# ── Construction du graphe ────────────────────────────────────
def build_graph():
    g = StateGraph(AgentState)
    g.add_node("supervisor",    supervisor_node)
    g.add_node("agent_coder",   agent_coder_node)
    g.add_node("agent_tech",    agent_tech_node)
    g.add_node("agent_human",   agent_human_node)
    g.add_node("agent_science", agent_science_node)
    g.add_node("agent_health",  agent_health_node)
    g.add_node("agent_general", agent_general_node)
    g.add_node("checker",       quality_checker_node)

    g.set_entry_point("supervisor")
    g.add_conditional_edges("supervisor", route_topic, {
        "coder":   "agent_coder",
        "tech":    "agent_tech",
        "human":   "agent_human",
        "science": "agent_science",
        "health":  "agent_health",
        "general": "agent_general",
    })
    for agent in ["agent_coder","agent_tech","agent_human","agent_science","agent_health","agent_general"]:
        g.add_edge(agent, "checker")
    g.add_conditional_edges("checker", route_quality, {"end": END, "retry": "supervisor"})
    return g.compile()

edubot_graph = build_graph()
print("Graphe LangGraph construit avec succes")

# ── Streaming ─────────────────────────────────────────────────
async def stream_response(text: str, meta: dict) -> AsyncIterator[str]:
    yield f"data: {json.dumps({'type': 'meta', **meta})}\n\n"
    await asyncio.sleep(0)
    words = text.split(" ")
    for i, word in enumerate(words):
        chunk = word + (" " if i < len(words) - 1 else "")
        yield f"data: {json.dumps({'type': 'chunk', 'text': chunk})}\n\n"
        await asyncio.sleep(0.015)
    yield f"data: {json.dumps({'type': 'done'})}\n\n"

# ── Application FastAPI ───────────────────────────────────────
app = FastAPI(
    title="EduBot",
    description="Assistant pedagogique professionnel - Janvier NZAMBIMANA, M1 ITN",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

class Message(BaseModel):
    role:    str
    content: str

class ChatRequest(BaseModel):
    message: str
    history: Optional[List[Message]] = []

class FeedbackRequest(BaseModel):
    message_id: str
    rating:     int  # 1-5
    comment:    Optional[str] = ""

def _initial_state(request: ChatRequest) -> AgentState:
    return {
        "question":    request.message,
        "history":     [{"role": m.role, "content": m.content} for m in request.history],
        "topic":       "general",
        "response":    "",
        "model_used":  "",
        "quality_ok":  False,
        "retry_count": 0,
        "app_type":    "",
        "code_plan":   [],
        "code_parts":  [],
        "language":    "fr",
    }

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    loop  = asyncio.get_event_loop()
    start = time.time()
    final = await loop.run_in_executor(None, edubot_graph.invoke, _initial_state(request))
    meta  = {
        "topic":        final["topic"],
        "model_used":   final["model_used"],
        "code_plan":    final.get("code_plan", []),
        "language":     final.get("language", "fr"),
        "elapsed_ms":   int((time.time() - start) * 1000),
    }
    return StreamingResponse(
        stream_response(final["response"], meta),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

@app.post("/chat")
async def chat_normal(request: ChatRequest):
    loop  = asyncio.get_event_loop()
    start = time.time()
    final = await loop.run_in_executor(None, edubot_graph.invoke, _initial_state(request))
    return {
        "response":   final["response"],
        "model_used": final["model_used"],
        "topic":      final["topic"],
        "code_plan":  final.get("code_plan", []),
        "language":   final.get("language", "fr"),
        "elapsed_ms": int((time.time() - start) * 1000),
    }

@app.post("/feedback")
async def submit_feedback(feedback: FeedbackRequest):
    # Stocker le feedback (a connecter a une base de donnees)
    print(f"Feedback recu - ID: {feedback.message_id}, Note: {feedback.rating}/5, Commentaire: {feedback.comment}")
    return {"status": "ok", "message": "Merci pour votre retour, cela aide a ameliorer EduBot."}

@app.get("/models")
def list_models():
    return {
        "models_available": [
            {"id": "llama-3.3-70b-versatile", "provider": "Groq", "usage": "tech, general, code"},
            {"id": "mixtral-8x7b-32768",       "provider": "Groq", "usage": "fallback general"},
            {"id": "gemini-2.0-flash-exp",     "provider": "Google", "usage": "humanites, sciences"},
            {"id": "gemini-1.5-flash",         "provider": "Google", "usage": "fallback rapide"},
            {"id": "gemini-1.5-pro",           "provider": "Google", "usage": "reponses complexes"},
        ],
        "routing": "Automatique selon le sujet detecte"
    }

@app.get("/")
def root():
    return {
        "status":  "EduBot en ligne",
        "version": "1.0.0",
        "author":  "Janvier NZAMBIMANA - M1 ITN",
        "topics":  ["tech", "science", "humanites", "sante", "general", "code"],
        "languages": ["francais", "anglais", "swahili", "kirundi"],
    }

@app.get("/health")
def health():
    return {"status": "ok", "timestamp": time.time()}

print("FastAPI configure - EduBot v1 pret")