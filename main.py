import os, json, asyncio, time
from typing import TypedDict, List, Optional, AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langgraph.graph import StateGraph, END
import groq as groq_lib
import google.generativeai as genai

# ════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════
GROQ_API_KEY   = os.getenv("GROQ_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

groq_client = groq_lib.Groq(api_key=GROQ_API_KEY)
genai.configure(api_key=GEMINI_API_KEY)

# ── Modèles Groq actifs (mixtral et llama3-70b-8192 dépréciés) ──
GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "meta-llama/llama-4-maverick-17b-128e-instruct",
    "qwen/qwen-3-32b",
    "gemma2-9b-it",
    "gemma-7b-it",
]

# ── Modèles Gemini actifs (1.5-flash et 1.5-pro retirés depuis avril 2025) ──
GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-2.0-flash",
]

# Cache des modèles Gemini initialisés
_gemini_cache = {}

def get_gemini_model(name: str):
    if name not in _gemini_cache:
        try:
            _gemini_cache[name] = genai.GenerativeModel(name)
        except Exception as e:
            print(f"  Impossible d'initialiser {name}: {e}")
            return None
    return _gemini_cache[name]

# ════════════════════════════════════════════════════════════
# MESSAGES D'ERREUR UTILISATEUR — jamais les vrais messages serveur
# ════════════════════════════════════════════════════════════
def user_friendly_error(raw_error: str) -> str:
    """
    Transforme les erreurs techniques en messages lisibles pour l'utilisateur.
    Ne jamais exposer les messages serveur bruts.
    """
    e = str(raw_error).lower()

    if any(k in e for k in ["rate limit", "429", "quota", "too many requests"]):
        return "Le service est momentanément surchargé. Veuillez réessayer dans quelques instants."

    if any(k in e for k in ["timeout", "timed out", "read timeout", "connect timeout"]):
        return "La requête a pris trop de temps. Vérifiez votre connexion et réessayez."

    if any(k in e for k in ["connection", "network", "unreachable", "refused", "socket"]):
        return "Impossible de joindre le service IA. Vérifiez votre connexion internet."

    if any(k in e for k in ["authentication", "api key", "unauthorized", "401", "403", "invalid_api_key"]):
        return "Le service IA est temporairement inaccessible. L'équipe technique a été notifiée."

    if any(k in e for k in ["overloaded", "503", "502", "500", "server error", "internal"]):
        return "Les serveurs sont surchargés en ce moment. Réessayez dans quelques secondes."

    if any(k in e for k in ["model not found", "model_not_found", "404", "deprecated"]):
        return "Un modèle IA est en cours de mise à jour. Tentative avec un autre modèle..."

    if any(k in e for k in ["context_length", "too long", "maximum context", "token"]):
        return "Votre message est trop long. Veuillez le raccourcir et réessayer."

    if any(k in e for k in ["content", "safety", "filtered", "blocked", "policy"]):
        return "Cette requête ne peut pas être traitée pour des raisons de sécurité."

    # Par défaut : message générique sans détails techniques
    return "Une erreur temporaire est survenue. Veuillez réessayer dans quelques instants."


# ════════════════════════════════════════════════════════════
# MOTEUR DE FALLBACK UNIVERSEL — cœur du système
# ════════════════════════════════════════════════════════════
def call_any_model(
    messages: list,
    max_tokens: int = 2048,
    temperature: float = 0.6,
    prefer_gemini: bool = False,
) -> tuple:
    """
    Essaie tous les modèles disponibles dans l'ordre.
    Retourne (texte_reponse, nom_modele_utilise).
    Ne lève JAMAIS d'exception — retourne un message d'erreur clair sinon.
    """

    raw_errors = []

    def try_groq_models():
        for model in GROQ_MODELS:
            try:
                print(f"  Essai Groq: {model}")
                resp = groq_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                text = resp.choices[0].message.content
                if text and len(text.strip()) > 10:
                    print(f"  ✓ Succès: {model}")
                    return text, f"Groq/{model}"
            except Exception as e:
                raw_errors.append(str(e))
                print(f"  ✗ Groq/{model}: {str(e)[:80]}")
        return None, None

    def try_gemini_models():
        prompt_parts = []
        for m in messages:
            role    = m.get("role", "user")
            content = m.get("content", "")
            if role == "system":
                prompt_parts.append(f"[Instructions]: {content}")
            elif role == "user":
                prompt_parts.append(f"Question: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Réponse précédente: {content}")
        full_prompt = "\n\n".join(prompt_parts)

        for model_name in GEMINI_MODELS:
            try:
                print(f"  Essai Gemini: {model_name}")
                model = get_gemini_model(model_name)
                if model is None:
                    continue
                resp = model.generate_content(
                    full_prompt,
                    generation_config=genai.types.GenerationConfig(
                        max_output_tokens=max_tokens,
                        temperature=temperature,
                    )
                )
                text = resp.text
                if text and len(text.strip()) > 10:
                    print(f"  ✓ Succès: {model_name}")
                    return text, f"Google/{model_name}"
            except Exception as e:
                raw_errors.append(str(e))
                print(f"  ✗ Gemini/{model_name}: {str(e)[:80]}")
        return None, None

    if prefer_gemini:
        sequences = [try_gemini_models, try_groq_models]
    else:
        sequences = [try_groq_models, try_gemini_models]

    for try_fn in sequences:
        text, model_name = try_fn()
        if text:
            return text, model_name

    # Tous les modèles ont échoué — message utilisateur sans détails techniques
    last_error = raw_errors[-1] if raw_errors else "unknown"
    friendly_msg = user_friendly_error(last_error)
    fallback_msg = (
        f"{friendly_msg}\n\n"
        "Si le problème persiste, contactez le support : janviernzambimana91@gmail.com"
    )
    print(f"  ✗✗ TOUS LES MODELES ONT ECHOUE. Dernière erreur brute : {last_error[:120]}")
    return fallback_msg, "service_indisponible"


def build_messages(system: str, history: list, question: str) -> list:
    msgs = [{"role": "system", "content": system}]
    for h in history[-8:]:
        msgs.append({"role": h["role"], "content": h["content"]})
    msgs.append({"role": "user", "content": question})
    return msgs

# ════════════════════════════════════════════════════════════
# PROMPTS & SIGNATURES
# ════════════════════════════════════════════════════════════
AUTHOR_SIGNATURE = """

---
*Réponse générée par EduBot — Assistant Pédagogique Intelligent*
*Janvier NZAMBIMANA | Étudiant M1 ITN*
*Support / signaler une erreur : janviernzambimana91@gmail.com*

*EduBot est gratuit pour tous les étudiants. Si utile, soutenez via :*
*Mobile Money / Lumicash : +257 68 58 97 29 (Janvier NZAMBIMANA)*

> **Important** : Vérifiez toujours les informations importantes avant de les utiliser."""

BASE_PROMPT = """Tu es EduBot, assistant pédagogique professionnel créé par Janvier NZAMBIMANA (M1 ITN).
Tu aides les étudiants africains et mondiaux dans leurs apprentissages.
Réponds en français sauf si l'utilisateur écrit dans une autre langue.
Sois clair, pédagogique, bienveillant, précis.
Structure tes réponses avec des titres quand utile.
N'utilise PAS d'emojis dans tes réponses."""

LANG_SUFFIXES = {
    "en": "\nThe user writes in English. Respond fully in English.",
    "sw": "\nMtumiaji anaandika kwa Kiswahili. Jibu kwa Kiswahili.",
    "rn": "\nUmukoresha andika mu Kirundi. Subiza mu Kirundi.",
    "fr": "",
}

def get_system_prompt(lang: str) -> str:
    return BASE_PROMPT + LANG_SUFFIXES.get(lang, "")

# ════════════════════════════════════════════════════════════
# DÉTECTION DE LANGUE
# ════════════════════════════════════════════════════════════
def detect_language(text: str) -> str:
    t = text.lower()
    w = t.split()
    en_words = {"the","is","are","what","how","why","when","who","i","you","we","please","help","can","my","do"}
    sw_words = {"ni","na","ya","kwa","katika","nini","jinsi","tafadhali","msaada","sema","wewe"}
    rn_words = {"ndi","ni","kuri","muri","ivyo","gute","ingene","murakoze","ndabashimiye","ubu"}
    en = sum(1 for x in w if x in en_words)
    sw = sum(1 for x in w if x in sw_words)
    rn = sum(1 for x in w if x in rn_words)
    if en > sw and en > rn and en >= 2: return "en"
    if sw > rn and sw >= 2:             return "sw"
    if rn >= 2:                         return "rn"
    return "fr"

# ════════════════════════════════════════════════════════════
# ÉTAT LANGGRAPH
# ════════════════════════════════════════════════════════════
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

# ════════════════════════════════════════════════════════════
# NŒUD 1 : SUPERVISEUR
# ════════════════════════════════════════════════════════════
def supervisor_node(state: AgentState) -> AgentState:
    q    = state["question"].lower()
    lang = detect_language(state["question"])

    app_triggers = [
        "cree une app","creer une application","code une app",
        "developpe une app","fais-moi une app","make an app",
        "create an app","build an app","cree un programme",
        "cree un site","cree un systeme","code complet",
        "application complete","projet complet",
        "cree un jeu","fais un jeu","fais une application",
        "construis une app","write a program","write an app",
        "create a website","build a website","faire une app",
    ]
    is_app = any(t in q for t in app_triggers)

    flutter_kw = ["flutter","dart","mobile","application mobile"]
    web_kw     = ["html","css","javascript","react","vue","angular","site web","frontend","webpage"]
    sql_kw     = ["sql","base de donnees","mysql","postgresql","sqlite","mongodb","database"]
    tech_kw    = ["code","programme","algorithme","fonction","boucle","classe","api",
                  "bug","erreur","reseau","linux","git","docker","machine learning",
                  "intelligence artificielle","ia","ml","data","cybersecurite"]
    science_kw = ["mathematiques","physique","chimie","equation","integrale","derivee",
                  "vecteur","biologie","atome","formule","statistiques","probabilites",
                  "thermodynamique","mecanique","optique","electricite"]
    human_kw   = ["histoire","philosophie","economie","politique","litterature",
                  "psychologie","sociologie","droit","colonialisme","revolution",
                  "geographie","culture","religion","ethique","anthropologie"]
    health_kw  = ["sante","maladie","medecine","symptome","traitement","nutrition",
                  "sport","bien-etre","mental","depression","anxiete","medicament"]

    if is_app:
        app_type = "python"
        if any(k in q for k in flutter_kw): app_type = "flutter"
        elif any(k in q for k in web_kw):   app_type = "web"
        elif any(k in q for k in sql_kw):   app_type = "sql"
        return {**state, "topic": "coder", "app_type": app_type, "language": lang}

    scores = {
        "tech":    sum(1 for k in tech_kw    if k in q),
        "science": sum(1 for k in science_kw if k in q),
        "human":   sum(1 for k in human_kw   if k in q),
        "health":  sum(1 for k in health_kw  if k in q),
    }
    best  = max(scores, key=scores.get)
    topic = best if scores[best] > 0 else "general"
    return {**state, "topic": topic, "app_type": "", "language": lang}

# ════════════════════════════════════════════════════════════
# NŒUD 2 : AGENT CODEUR
# ════════════════════════════════════════════════════════════
def agent_coder_node(state: AgentState) -> AgentState:
    q        = state["question"]
    app_type = state.get("app_type", "python")
    lang     = state.get("language", "fr")
    sys_p    = get_system_prompt(lang)

    print(f"\n[Agent Codeur] app_type={app_type}, lang={lang}")

    # ── Étape A : Plan JSON ──
    plan_prompt_msgs = [{
        "role": "user",
        "content": (
            f'Tu es expert développeur. Demande: "{q}"\n'
            f'Réponds UNIQUEMENT avec ce JSON (pas de texte, pas de markdown) :\n'
            f'{{"app_name":"...","description":"...","technology":"{app_type}",'
            f'"etapes":["Etape 1: ...","Etape 2: ...","Etape 3: ...","Etape 4: ..."],'
            f'"fichiers":["fichier1"],"dependances":["dep1"]}}'
        )
    }]

    code_plan = ["Analyse des besoins", "Architecture", "Développement", "Tests"]
    try:
        raw, _ = call_any_model(plan_prompt_msgs, max_tokens=600, temperature=0.1)
        raw    = raw.replace("```json","").replace("```","").strip()
        s, e   = raw.find("{"), raw.rfind("}") + 1
        if s != -1 and e > s:
            plan_data = json.loads(raw[s:e])
            code_plan = plan_data.get("etapes", code_plan)
            print(f"  Plan JSON OK: {len(code_plan)} étapes")
    except Exception as ex:
        print(f"  Plan JSON erreur (ignorée): {ex}")

    # ── Étape B : Code complet ──
    lang_code = {"flutter": "dart", "web": "html", "sql": "sql"}.get(app_type, "python")

    code_msgs = [
        {"role": "system", "content": sys_p},
        {"role": "user",   "content": (
            f'L\'étudiant demande : "{q}"\n'
            f'Technologie : {app_type}\n\n'
            f'Génère le CODE COMPLET professionnel :\n\n'
            f'## Description du projet\n[2-3 phrases]\n\n'
            f'## Plan de développement\n[étapes numérotées]\n\n'
            f'## Prérequis et installation\n[commandes pip/npm/etc.]\n\n'
            f'## Code complet\n\n'
            f'### Fichier : [nom.extension]\n```{lang_code}\n[code complet commenté]\n```\n\n'
            f'## Comment exécuter\n[instructions précises]\n\n'
            f'## Améliorations possibles\n[3 idées]\n\n'
            f'Le code doit être 100% fonctionnel, bien commenté, sans emojis.'
        )}
    ]

    print(f"  Génération du code...")
    full_code, model_used = call_any_model(
        code_msgs, max_tokens=4096, temperature=0.4, prefer_gemini=False
    )

    return {
        **state,
        "response":   full_code + AUTHOR_SIGNATURE,
        "model_used": f"{model_used} (Codeur)",
        "code_plan":  code_plan,
    }

# ════════════════════════════════════════════════════════════
# NŒUDS 3a–3e : AGENTS SPÉCIALISÉS
# ════════════════════════════════════════════════════════════
def _agent_respond(state: AgentState, extra_instruction: str,
                   label: str, prefer_gemini: bool = False) -> AgentState:
    lang   = state.get("language", "fr")
    sys_p  = get_system_prompt(lang) + "\n\n" + extra_instruction
    msgs   = build_messages(sys_p, state["history"], state["question"])
    print(f"\n[{label}] Appel modèles...")
    r, model = call_any_model(msgs, max_tokens=2048, temperature=0.6,
                               prefer_gemini=prefer_gemini)
    return {**state, "response": r + AUTHOR_SIGNATURE, "model_used": f"{model} ({label})"}

def agent_tech_node(state):
    return _agent_respond(state,
        "Tu es expert en informatique, programmation et technologies numériques. "
        "Réponds de façon structurée avec exemples concrets. Sans emojis.",
        "Tech", prefer_gemini=False)

def agent_human_node(state):
    return _agent_respond(state,
        "Tu es expert en sciences humaines : histoire, philosophie, économie, droit, géographie. "
        "Fournis des analyses nuancées avec contexte historique. Sans emojis.",
        "Humanités", prefer_gemini=True)

def agent_science_node(state):
    return _agent_respond(state,
        "Tu es expert en sciences exactes : maths, physique, chimie, biologie. "
        "Explique avec rigueur, montre les formules et calculs étape par étape. Sans emojis.",
        "Sciences", prefer_gemini=True)

def agent_health_node(state):
    return _agent_respond(state,
        "Tu es assistant santé éducatif. Donne des infos générales sur santé et bien-être. "
        "RAPPELLE TOUJOURS de consulter un professionnel de santé pour tout problème médical. Sans emojis.",
        "Santé", prefer_gemini=False)

def agent_general_node(state):
    return _agent_respond(state,
        "Réponds de manière claire, complète et pédagogique. Sans emojis.",
        "General", prefer_gemini=False)

# ════════════════════════════════════════════════════════════
# NŒUD 4 : VÉRIFICATEUR DE QUALITÉ
# ════════════════════════════════════════════════════════════
def quality_checker_node(state: AgentState) -> AgentState:
    r     = state.get("response", "")
    retry = state.get("retry_count", 0)
    model = state.get("model_used", "")

    is_service_down = model == "service_indisponible"
    is_too_short    = len(r.strip()) < 80

    # Si le service est indisponible, inutile de retenter indéfiniment
    ok = not is_too_short
    if is_service_down:
        ok = True  # on accepte le message d'erreur poli

    if not ok and retry < 2:
        print(f"  [Checker] Qualité insuffisante, retry {retry+1}/2")
        return {**state, "quality_ok": False, "retry_count": retry + 1}

    print(f"  [Checker] OK (longueur={len(r)}, modèle={model})")
    return {**state, "quality_ok": True}

# ════════════════════════════════════════════════════════════
# ROUTEURS
# ════════════════════════════════════════════════════════════
def route_topic(state):   return state["topic"]
def route_quality(state): return "end" if state["quality_ok"] else "retry"

# ════════════════════════════════════════════════════════════
# CONSTRUCTION DU GRAPHE
# ════════════════════════════════════════════════════════════
def build_graph():
    g = StateGraph(AgentState)
    nodes = {
        "supervisor":    supervisor_node,
        "agent_coder":   agent_coder_node,
        "agent_tech":    agent_tech_node,
        "agent_human":   agent_human_node,
        "agent_science": agent_science_node,
        "agent_health":  agent_health_node,
        "agent_general": agent_general_node,
        "checker":       quality_checker_node,
    }
    for name, fn in nodes.items():
        g.add_node(name, fn)

    g.set_entry_point("supervisor")
    g.add_conditional_edges("supervisor", route_topic, {
        "coder":   "agent_coder",
        "tech":    "agent_tech",
        "human":   "agent_human",
        "science": "agent_science",
        "health":  "agent_health",
        "general": "agent_general",
    })
    for agent in ["agent_coder","agent_tech","agent_human",
                  "agent_science","agent_health","agent_general"]:
        g.add_edge(agent, "checker")
    g.add_conditional_edges("checker", route_quality,
                            {"end": END, "retry": "supervisor"})
    return g.compile()

edubot_graph = build_graph()
print("Graphe LangGraph construit avec succès")

# ════════════════════════════════════════════════════════════
# STREAMING SSE — mot par mot avec délai naturel
# ════════════════════════════════════════════════════════════
async def stream_response(text: str, meta: dict) -> AsyncIterator[str]:
    yield f"data: {json.dumps({'type': 'meta', **meta})}\n\n"
    await asyncio.sleep(0)

    words = text.split(" ")
    for i, word in enumerate(words):
        chunk = word + (" " if i < len(words) - 1 else "")
        yield f"data: {json.dumps({'type': 'chunk', 'text': chunk})}\n\n"
        # Délai naturel : ponctuation = petite pause, sinon rythme régulier
        if word.endswith((".", "!", "?", ":", ";")):
            await asyncio.sleep(0.055)
        elif word.endswith(","):
            await asyncio.sleep(0.030)
        else:
            await asyncio.sleep(0.022)

    yield f"data: {json.dumps({'type': 'done'})}\n\n"

# ════════════════════════════════════════════════════════════
# API FASTAPI
# ════════════════════════════════════════════════════════════
app = FastAPI(
    title="EduBot",
    description="Assistant pédagogique — Janvier NZAMBIMANA, M1 ITN",
    version="3.0.0",
)
app.add_middleware(CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class Message(BaseModel):
    role:    str
    content: str

class ChatRequest(BaseModel):
    message: str
    history: Optional[List[Message]] = []

class FeedbackRequest(BaseModel):
    message_id: str
    rating:     int
    comment:    Optional[str] = ""

def _initial_state(req: ChatRequest) -> AgentState:
    return {
        "question":    req.message,
        "history":     [{"role": m.role, "content": m.content} for m in req.history],
        "topic":       "general", "response":    "",
        "model_used":  "",        "quality_ok":  False,
        "retry_count": 0,         "app_type":    "",
        "code_plan":   [],        "code_parts":  [],
        "language":    "fr",
    }

@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    loop  = asyncio.get_event_loop()
    start = time.time()
    final = await loop.run_in_executor(None, edubot_graph.invoke, _initial_state(req))
    meta  = {
        "topic":      final["topic"],
        "model_used": final["model_used"],
        "code_plan":  final.get("code_plan", []),
        "language":   final.get("language", "fr"),
        "elapsed_ms": int((time.time() - start) * 1000),
    }
    return StreamingResponse(
        stream_response(final["response"], meta),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

@app.post("/chat")
async def chat_normal(req: ChatRequest):
    loop  = asyncio.get_event_loop()
    start = time.time()
    final = await loop.run_in_executor(None, edubot_graph.invoke, _initial_state(req))
    return {
        "response":   final["response"],
        "model_used": final["model_used"],
        "topic":      final["topic"],
        "code_plan":  final.get("code_plan", []),
        "language":   final.get("language", "fr"),
        "elapsed_ms": int((time.time() - start) * 1000),
    }

@app.post("/feedback")
async def submit_feedback(fb: FeedbackRequest):
    print(f"Feedback — ID:{fb.message_id} Note:{fb.rating}/5 : {fb.comment}")
    return {"status": "ok", "message": "Merci pour votre retour !"}

@app.get("/models")
def list_models():
    return {
        "groq_models":   GROQ_MODELS,
        "gemini_models": GEMINI_MODELS,
        "total":         len(GROQ_MODELS) + len(GEMINI_MODELS),
        "strategy":      "Fallback automatique : Groq d'abord, puis Gemini (ou inverse selon le topic)",
        "note":          "mixtral-8x7b et llama3-70b-8192 retirés (dépréciés). gemini-1.5 retiré depuis avril 2025.",
    }

@app.get("/")
def root():
    return {
        "status":    "EduBot en ligne",
        "version":   "3.0.0",
        "author":    "Janvier NZAMBIMANA — M1 ITN",
        "topics":    ["tech","science","humanités","santé","général","code"],
        "languages": ["français","anglais","swahili","kirundi"],
        "models":    len(GROQ_MODELS) + len(GEMINI_MODELS),
    }

@app.get("/health")
def health():
    return {"status": "ok", "timestamp": time.time(), "version": "3.0.0"}

print(f"EduBot v3 prêt — {len(GROQ_MODELS) + len(GEMINI_MODELS)} modèles disponibles")
