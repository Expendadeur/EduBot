import os, json, asyncio, time, threading, io
from typing import TypedDict, List, Optional, AsyncIterator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel
from langgraph.graph import StateGraph, END
import groq as groq_lib
import google.generativeai as genai

import firebase_admin
from firebase_admin import credentials, firestore

# ════════════════════════════════════════════════════════════
# CONFIGURATION FIREBASE ADMIN SDK
# ════════════════════════════════════════════════════════════
db = None
try:
    cred_path = os.getenv("FIREBASE_CREDENTIALS_PATH", "Edubot/android/app/edubot-a6cc2-firebase-adminsdk-fbsvc-5b57c223ab.json")
    if not os.path.exists(cred_path):
        cred_path = "edubot-a6cc2-firebase-adminsdk-fbsvc-5b57c223ab.json"
    
    if os.path.exists(cred_path):
        cred = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(cred)
        db = firestore.client()
        print(f"[FIREBASE] Firestore initialisé avec succès ({cred_path})")
    elif os.getenv("FIREBASE_CONFIG_JSON"):
        cred_json = json.loads(os.getenv("FIREBASE_CONFIG_JSON"))
        cred = credentials.Certificate(cred_json)
        firebase_admin.initialize_app(cred)
        db = firestore.client()
        print("[FIREBASE] Firestore initialisé depuis FIREBASE_CONFIG_JSON")
    else:
        print("[FIREBASE] Avertissement: Fichier de clés Firebase introuvable.")
except Exception as e:
    print(f"[FIREBASE] Erreur d'initialisation Firestore: {e}")

# ════════════════════════════════════════════════════════════
# CONFIGURATION IA
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


# ════════════════════════════════════════════════════════════
# STREAMING RÉEL — les tokens sont envoyés au client au fur et à
# mesure qu'ils arrivent du modèle (Groq stream=True / Gemini
# stream=True). Aucun texte n'est généré à l'avance puis rejoué
# avec un délai artificiel : ce qui part sur le fil est ce que le
# modèle est réellement en train d'écrire.
# ════════════════════════════════════════════════════════════
async def real_stream(messages: list, max_tokens: int = 2048,
                       temperature: float = 0.6, prefer_gemini: bool = False):
    loop  = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue()
    model_used = {"name": None}

    def _try_groq_stream():
        for model in GROQ_MODELS:
            try:
                print(f"  [stream] Essai Groq: {model}")
                stream = groq_client.chat.completions.create(
                    model=model, messages=messages,
                    max_tokens=max_tokens, temperature=temperature, stream=True,
                )
                collected = ""
                for chunk in stream:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        collected += delta
                        loop.call_soon_threadsafe(queue.put_nowait, ("chunk", delta))
                if len(collected.strip()) > 10:
                    print(f"  [stream] ✓ Succès: {model}")
                    return f"Groq/{model}"
            except Exception as e:
                print(f"  [stream] ✗ Groq/{model}: {str(e)[:80]}")
                continue
        return None

    def _try_gemini_stream():
        prompt_parts = []
        for m in messages:
            role, content = m.get("role", "user"), m.get("content", "")
            if role == "system":
                prompt_parts.append(f"[Instructions]: {content}")
            elif role == "user":
                prompt_parts.append(f"Question: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Réponse précédente: {content}")
        full_prompt = "\n\n".join(prompt_parts)

        for model_name in GEMINI_MODELS:
            try:
                print(f"  [stream] Essai Gemini: {model_name}")
                model = get_gemini_model(model_name)
                if model is None:
                    continue
                resp_stream = model.generate_content(
                    full_prompt,
                    generation_config=genai.types.GenerationConfig(
                        max_output_tokens=max_tokens, temperature=temperature,
                    ),
                    stream=True,
                )
                collected = ""
                for chunk in resp_stream:
                    if chunk.text:
                        collected += chunk.text
                        loop.call_soon_threadsafe(queue.put_nowait, ("chunk", chunk.text))
                if len(collected.strip()) > 10:
                    print(f"  [stream] ✓ Succès: {model_name}")
                    return f"Google/{model_name}"
            except Exception as e:
                print(f"  [stream] ✗ Gemini/{model_name}: {str(e)[:80]}")
                continue
        return None

    def producer():
        try:
            sequences = [_try_gemini_stream, _try_groq_stream] if prefer_gemini \
                        else [_try_groq_stream, _try_gemini_stream]
            for fn in sequences:
                name = fn()
                if name:
                    model_used["name"] = name
                    return
            # Tous les modèles ont échoué en streaming
            friendly = user_friendly_error("tous les modeles ont echoue")
            fallback = f"{friendly}\n\nSi le problème persiste, contactez le support : janviernzambimana91@gmail.com"
            loop.call_soon_threadsafe(queue.put_nowait, ("chunk", fallback))
            model_used["name"] = "service_indisponible"
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, ("done", None))

    threading.Thread(target=producer, daemon=True).start()

    yield f"data: {json.dumps({'type': 'meta', 'status': 'start'})}\n\n"
    while True:
        kind, payload = await queue.get()
        if kind == "chunk":
            yield f"data: {json.dumps({'type': 'chunk', 'text': payload})}\n\n"
        else:
            yield f"data: {json.dumps({'type': 'done', 'model_used': model_used['name']})}\n\n"
            break


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
*Réponse générée par EduBot — Assistant Pédagogique Intelligent & Module EMI*
> **Important** : Vérifiez toujours les informations importantes avant de les utiliser."""

BASE_PROMPT = """Tu es EduBot, assistant pédagogique professionnel et expert en éducation aux médias et à l'information (EMI).
Tu aides les étudiants et les citoyens dans leurs apprentissages et la vérification d'informations.
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
APP_TRIGGERS = [
    "cree une app","creer une application","code une app",
    "developpe une app","fais-moi une app","make an app",
    "create an app","build an app","cree un programme",
    "cree un site","cree un systeme","code complet",
    "application complete","projet complet",
    "cree un jeu","fais un jeu","fais une application",
    "construis une app","write a program","write an app",
    "create a website","build a website","faire une app",
]
FLUTTER_KW = ["flutter","dart","mobile","application mobile"]
WEB_KW     = ["html","css","javascript","react","vue","angular","site web","frontend","webpage"]
SQL_KW     = ["sql","base de donnees","mysql","postgresql","sqlite","mongodb","database"]
TECH_KW    = ["code","programme","algorithme","fonction","boucle","classe","api",
              "bug","erreur","reseau","linux","git","docker","machine learning",
              "intelligence artificielle","ia","ml","data","cybersecurite"]
SCIENCE_KW = ["mathematiques","physique","chimie","equation","integrale","derivee",
              "vecteur","biologie","atome","formule","statistiques","probabilites",
              "thermodynamique","mecanique","optique","electricite"]
HUMAN_KW   = ["histoire","philosophie","economie","politique","litterature",
              "psychologie","sociologie","droit","colonialisme","revolution",
              "geographie","culture","religion","ethique","anthropologie"]
HEALTH_KW  = ["sante","maladie","medecine","symptome","traitement","nutrition",
              "sport","bien-etre","mental","depression","anxiete","medicament"]

def classify_question(question: str) -> dict:
    """
    Classification partagée par le graphe LangGraph (endpoint /chat) et le
    endpoint de streaming réel /chat/stream, pour éviter toute divergence
    entre les deux chemins.
    """
    q    = question.lower()
    lang = detect_language(question)
    is_app = any(t in q for t in APP_TRIGGERS)

    if is_app:
        app_type = "python"
        if any(k in q for k in FLUTTER_KW): app_type = "flutter"
        elif any(k in q for k in WEB_KW):   app_type = "web"
        elif any(k in q for k in SQL_KW):   app_type = "sql"
        return {"topic": "coder", "app_type": app_type, "language": lang}

    scores = {
        "tech":    sum(1 for k in TECH_KW    if k in q),
        "science": sum(1 for k in SCIENCE_KW if k in q),
        "human":   sum(1 for k in HUMAN_KW   if k in q),
        "health":  sum(1 for k in HEALTH_KW  if k in q),
    }
    best  = max(scores, key=scores.get)
    topic = best if scores[best] > 0 else "general"
    return {"topic": topic, "app_type": "", "language": lang}

TOPIC_INSTRUCTIONS = {
    "tech": ("Tu es expert en informatique, programmation et technologies numériques. "
             "Réponds de façon structurée avec exemples concrets. Sans emojis.", False),
    "science": ("Tu es expert en sciences exactes : maths, physique, chimie, biologie. "
                "Explique avec rigueur, montre les formules et calculs étape par étape. Sans emojis.", True),
    "human": ("Tu es expert en sciences humaines : histoire, philosophie, économie, droit, géographie. "
              "Fournis des analyses nuancées avec contexte historique. Sans emojis.", True),
    "health": ("Tu es assistant santé éducatif. Donne des infos générales sur santé et bien-être. "
               "RAPPELLE TOUJOURS de consulter un professionnel de santé pour tout problème médical. Sans emojis.", False),
    "general": ("Réponds de manière claire, complète et pédagogique. Sans emojis.", False),
}

def supervisor_node(state: AgentState) -> AgentState:
    classification = classify_question(state["question"])
    return {**state, **classification}

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
# API FASTAPI
# ════════════════════════════════════════════════════════════
app = FastAPI(
    title="EduBot",
    description="Assistant pédagogique — Janvier NZAMBIMANA, M1 ITN",
    version="3.1.0",
)
app.add_middleware(CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ════════════════════════════════════════════════════════════
# MODULE EMI — Éducation aux Médias et à l'Information
# (UNESCO Youth Hackathon 2026)
# Contenu réel (tips, sources, quiz) servi par le backend :
# une seule source de vérité, partagée par le front web et
# l'app Flutter — rien n'est codé en dur côté client.
# ════════════════════════════════════════════════════════════
EMI_FACTCHECK_SYSTEM_TEMPLATE = (
    "Tu es un expert en fact-checking et en éducation aux médias et à l'information (EMI) pour le pays : {country_name}. "
    "Formé aux méthodes utilisées par les grands réseaux de vérification indépendants dans le monde "
    "(Burundi Check, Africa Check, AFP Factuel, Congo Check, PesaCheck, etc.). "
    "Tu analyses une information de façon rigoureuse, nuancée, et spécifique au contexte local de ce pays ({country_name}). "
    "Tu réponds UNIQUEMENT avec un objet JSON valide, sans texte autour, "
    "sans balises markdown, exactement au format suivant :\n"
    '{"verdict":"VRAI|FAUX|PARTIELLEMENT VRAI|NON VÉRIFIABLE",'
    '"titre":"résumé du verdict en une phrase courte",'
    '"explication":"2 à 3 sentences expliquant le raisonnement et le contexte spécifique à ce pays ({country_name}) si applicable, sinon général.",'
    '"coach":"Explication pédagogique du Coach IA : 1 à 2 phrases simples expliquant POURQUOI cette information est peu fiable ou trompeuse, adaptée à un élève de lycée.",'
    '"conseils":"1 à 2 conseils pratiques pour vérifier ce type d\'information soi-même dans ce contexte quotidien.",'
    '"sources":"2 à 4 organismes CONCRETS et vérifiables, cités par leur nom réel"}\n\n'
    "Règle importante pour le champ \"sources\" : ne réponds JAMAIS par une catégorie vague "
    "comme \"un site de fact-checking\". Cite des noms précis, et en priorité les organismes locaux et régionaux "
    "pertinents pour {country_name} (ex: Burundi Check, ABP pour le Burundi, Congo Check pour la RDC, RBA pour le Rwanda, PesaCheck, etc.)."
)

def build_factcheck_messages(text: str, country_name: str) -> list:
    system_prompt = EMI_FACTCHECK_SYSTEM_TEMPLATE.format(country_name=country_name)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f'Information à analyser : "{text}"'},
    ]

EMI_TIPS = [
    {"icon": "ti-clock-pause", "color": "#F79009", "bg": "#FFFAEB",
     "title": "Respire avant de partager",
     "text": "Une info qui pousse à réagir tout de suite (peur, colère) est souvent conçue pour être partagée sans vérification."},
    {"icon": "ti-user-search", "color": "#2251CC", "bg": "#EEF3FF",
     "title": "Identifie la source réelle",
     "text": "Un message WhatsApp ou Facebook forwardé n'a pas de source vérifiable : remonte jusqu'au média ou à l'institution d'origine."},
    {"icon": "ti-calendar-off", "color": "#7C3AED", "bg": "#F5F3FF",
     "title": "Vérifie la date",
     "text": "Une vraie information republiée hors contexte, des années après, devient une fausse information dans le présent."},
    {"icon": "ti-photo-search", "color": "#17B26A", "bg": "#ECFDF3",
     "title": "Recherche l'image inversée",
     "text": "Une capture d'écran ou une photo peut venir d'un autre pays ou d'un autre événement : utilise la recherche d'image inversée."},
    {"icon": "ti-building-bank", "color": "#F04438", "bg": "#FFF0F0",
     "title": "Méfie-toi des fausses annonces officielles",
     "text": "Un gouvernement ou une banque n'annonce jamais une décision uniquement via une image WhatsApp non signée."},
    {"icon": "ti-wifi-off", "color": "#2251CC", "bg": "#EEF3FF",
     "title": "Le coût des données joue contre toi",
     "text": "En zone à connexion coûteuse, on partage vite sans ouvrir le lien : ouvre l'article avant de transférer, même si ça coûte des Mo."},
    {"icon": "ti-radio", "color": "#F79009", "bg": "#FFFAEB",
     "title": "Croise radio, presse écrite et web",
     "text": "Une info réellement importante circule sur plusieurs médias indépendants, pas uniquement dans un seul groupe WhatsApp."},
    {"icon": "ti-language", "color": "#7C3AED", "bg": "#F5F3FF",
     "title": "Attention aux traductions approximatives",
     "text": "Une citation traduite depuis une langue étrangère peut perdre son sens d'origine : cherche la déclaration originale."},
    {"icon": "ti-coin", "color": "#17B26A", "bg": "#ECFDF3",
     "title": "Doute des promesses trop belles",
     "text": "Argent gratuit, forfait internet illimité offert, cadeau d'une entreprise : ce sont des schémas classiques d'arnaque ou de désinformation."},
    {"icon": "ti-messages", "color": "#F04438", "bg": "#FFF0F0",
     "title": "Signale sans humilier",
     "text": "Si un proche partage une fausse info, corrige-le avec une source fiable en message privé plutôt que publiquement."},
]

EMI_SOURCES = [
    # Burundi
    {"name": "Burundi Check", "url": "https://burundicheck.org",
     "desc": "Initiative burundaise indépendante et pionnière de vérification des faits et de lutte contre les rumeurs.",
     "icon": "ti-shield-check", "color": "#F04438",
     "regions": ["BI", "afrique_est"], "topics": ["general", "politique", "sante"]},
    {"name": "ABP (Agence Burundaise de Presse)", "url": "http://abp.info.bi",
     "desc": "Agence officielle burundaise fournissant des informations étatiques et générales vérifiées.",
     "icon": "ti-news", "color": "#17B26A",
     "regions": ["BI", "afrique_est"], "topics": ["general", "politique"]},
    {"name": "Journal Iwacu Burundi", "url": "https://www.iwacu-burundi.org",
     "desc": "Principal groupe de presse indépendant du Burundi, reconnu pour ses enquêtes et sa vérification de l'info.",
     "icon": "ti-news", "color": "#2251CC",
     "regions": ["BI", "afrique_est"], "topics": ["general", "politique"]},

    # Régionaux & Globaux
    {"name": "Africa Check", "url": "https://africacheck.org",
     "desc": "Premier site de fact-checking indépendant en Afrique, vérifie les affirmations publiques et virales.",
     "icon": "ti-shield-check", "color": "#17B26A",
     "regions": ["global", "afrique_australe", "afrique_ouest", "afrique_est", "afrique_centrale"], "topics": ["general", "politique", "sante"]},
    {"name": "AFP Factuel", "url": "https://factuel.afp.com",
     "desc": "Cellule de vérification de l'Agence France-Presse, spécialisée dans les fausses images et vidéos virales.",
     "icon": "ti-news", "color": "#2251CC",
     "regions": ["global"], "topics": ["general", "politique", "viral"]},
    {"name": "OMS", "url": "https://www.who.int",
     "desc": "Organisation Mondiale de la Santé — source de référence pour toute information médicale ou sanitaire.",
     "icon": "ti-heartbeat", "color": "#F04438",
     "regions": ["global"], "topics": ["sante"]},
    {"name": "UNESCO", "url": "https://www.unesco.org",
     "desc": "Ressources et programmes d'éducation aux médias et à l'information (EMI) à l'échelle mondiale.",
     "icon": "ti-school", "color": "#7C3AED",
     "regions": ["global"], "topics": ["education", "general"]},
    {"name": "BBC Afrique", "url": "https://www.bbc.com/afrique",
     "desc": "Rédaction BBC dédiée à l'actualité africaine, avec un service régulier de vérification des faits.",
     "icon": "ti-broadcast", "color": "#F79009",
     "regions": ["global", "afrique_ouest", "afrique_est", "afrique_australe"], "topics": ["general", "politique"]},
    {"name": "RFI", "url": "https://www.rfi.fr",
     "desc": "Radio France Internationale — couverture et analyse continue de l'actualité africaine et internationale.",
     "icon": "ti-radio", "color": "#2251CC",
     "regions": ["global", "afrique_ouest", "afrique_centrale"], "topics": ["general", "politique"]},
    {"name": "PesaCheck", "url": "https://pesacheck.org",
     "desc": "Réseau de fact-checking dédié à l'Afrique de l'Est (Kenya, Tanzanie, Ouganda, Rwanda, Burundi...).",
     "icon": "ti-shield-check", "color": "#17B26A",
     "regions": ["afrique_est", "RW", "BI"], "topics": ["general", "politique", "sante"]},
    {"name": "Dubawa", "url": "https://dubawa.org",
     "desc": "Organisation de fact-checking couvrant le Nigeria, le Ghana, la Sierra Leone et le Liberia.",
     "icon": "ti-shield-check", "color": "#17B26A",
     "regions": ["afrique_ouest"], "topics": ["general", "politique"]},
    {"name": "Congo Check", "url": "https://congocheck.net",
     "desc": "Vérification des faits centrée sur la République Démocratique du Congo et l'Afrique centrale.",
     "icon": "ti-shield-check", "color": "#17B26A",
     "regions": ["afrique_centrale", "CD"], "topics": ["general", "politique"]},
    {"name": "Real411", "url": "https://www.real411.org",
     "desc": "Plateforme sud-africaine de signalement et de vérification de la désinformation.",
     "icon": "ti-shield-check", "color": "#17B26A",
     "regions": ["afrique_australe"], "topics": ["general", "politique"]},
    {"name": "Reuters Fact Check", "url": "https://www.reuters.com/fact-check",
     "desc": "Cellule de vérification de l'agence Reuters, couverture internationale des rumeurs virales.",
     "icon": "ti-news", "color": "#2251CC",
     "regions": ["global"], "topics": ["general", "politique", "viral"]},
    {"name": "CDC", "url": "https://www.cdc.gov",
     "desc": "Centres américains de contrôle des maladies — référence pour les questions de santé publique.",
     "icon": "ti-heartbeat", "color": "#F04438",
     "regions": ["global"], "topics": ["sante"]},
]

EMI_SUGGESTED_SOURCES: list = []

EMI_QUIZ = [
    {"q": "Un message WhatsApp affirme qu'un remède maison guérit une maladie grave. Que fais-tu en premier ?",
     "opts": ["Je le partage à ma famille par précaution", "Je vérifie sur le site de l'OMS ou avec un professionnel de santé",
              "Je le crois car il vient d'un proche", "Je l'ignore sans vérifier ni informer personne"],
     "ans": 1, "exp": "Une information de santé virale doit toujours être confrontée à une source médicale fiable comme l'OMS avant d'être crue ou partagée."},
    {"q": "Une image choc circule sur les réseaux sociaux pour illustrer un conflit récent. Quel est le meilleur réflexe ?",
     "opts": ["La partager immédiatement vu son caractère urgent", "Faire une recherche d'image inversée pour vérifier son origine",
              "Vérifier uniquement le nombre de partages", "Supposer qu'elle est vraie si elle a beaucoup de likes"],
     "ans": 1, "exp": "La recherche d'image inversée permet souvent de découvrir qu'une photo choc vient d'un autre lieu ou d'une autre époque."},
    {"q": "Qu'est-ce que la désinformation, par opposition à la mésinformation ?",
     "opts": ["Les deux termes sont identiques", "La désinformation est diffusée intentionnellement pour tromper",
              "La désinformation est toujours produite par un gouvernement", "La mésinformation est toujours volontaire"],
     "ans": 1, "exp": "La désinformation implique une intention de tromper, contrairement à la mésinformation qui est un partage d'erreur de bonne foi."},
    {"q": "Un compte anonyme prétend être un média officiel et annonce une mesure gouvernementale surprenante. Que dois-tu vérifier ?",
     "opts": ["Le nombre d'abonnés du compte", "L'existence d'une annonce correspondante sur le site officiel ou un média reconnu",
              "Si le message est bien orthographié", "Rien, un compte avec un joli logo est fiable"],
     "ans": 1, "exp": "Un nom ou un logo officiel peut être imité facilement : seule une source officielle vérifiable confirme une annonce publique."},
    {"q": "Pourquoi la date de publication d'un article est-elle importante en EMI ?",
     "opts": ["Elle ne l'est pas si le contenu semble vrai", "Un vieux fait relayé comme actuel peut créer une fausse impression du présent",
              "Seul le titre compte", "La date sert uniquement pour le classement du site"],
     "ans": 1, "exp": "Une information vraie mais ancienne, repartagée hors contexte, devient trompeuse car elle laisse croire à un événement récent."},
    {"q": "Quel est le rôle principal d'un site comme Africa Check ou AFP Factuel ?",
     "opts": ["Créer du contenu viral", "Vérifier des affirmations publiques et publier le résultat avec les preuves",
              "Remplacer les médias traditionnels", "Censurer les réseaux sociaux"],
     "ans": 1, "exp": "Ces organisations vérifient méthodiquement des affirmations virales ou publiques et publient leurs sources et leur méthode."},
    {"q": "Tu reçois un message annonçant un cadeau gratuit d'une grande entreprise si tu cliques sur un lien et le partages à 10 contacts. C'est probablement :",
     "opts": ["Une vraie promotion à saisir vite", "Une arnaque ou une chaîne de désinformation classique",
              "Un test officiel de l'entreprise", "Un cadeau garanti par WhatsApp"],
     "ans": 1, "exp": "La demande de partage massif avant toute vérification est une caractéristique classique des arnaques et chaînes virales."},
    {"q": "Une vidéo montre un homme politique tenant des propos choquants, mais le mouvement de ses lèvres est décalé et son teint semble étrange. Qu'est-ce que cela peut être ?",
     "opts": ["Un problème technique d'antenne", "Un deepfake (trucage vidéo par IA) visant à manipuler l'opinion",
              "Une mauvaise traduction", "Une preuve absolue de ses propos"],
     "ans": 1, "exp": "Les deepfakes présentent souvent des anomalies : décalage labial, clignement des yeux inexistant, flous ou incohérences au niveau du cou."},
    {"q": "Comment identifier une image suspectée d'être générée par une Intelligence Artificielle (Midjourney, DALL-E) ?",
     "opts": ["En observant les détails fins : doigts en surnombre, écritures floues, arrière-plans bizarres ou reflets impossibles",
              "En vérifiant si elle a plus de 1000 partages", "Toutes les images d'IA sont en noir et blanc", "Les images générées par IA sont parfaites et indétectables"],
     "ans": 0, "exp": "Les IA ont du mal à générer des détails complexes cohérents, notamment les mains (doigts collés/supplémentaires) et les textes."},
]

# Données des écoles partenaires et défis (Hackathon EMI 2026)
EMI_SCHOOLS = [
    {"name": "Lycée de Bujumbura", "city": "Bujumbura", "country": "BI", "score": 92.5, "participants": 120},
    {"name": "Lycée Notre Dame de Rohero", "city": "Bujumbura", "country": "BI", "score": 89.8, "participants": 95},
    {"name": "Lycée du Saint-Esprit", "city": "Bujumbura", "country": "BI", "score": 88.4, "participants": 140},
    {"name": "Lycée de Gitega", "city": "Gitega", "country": "BI", "score": 85.0, "participants": 110},
    {"name": "Green Hills Academy", "city": "Kigali", "country": "RW", "score": 91.2, "participants": 80},
    {"name": "Lycée Shaumba", "city": "Kinshasa", "country": "CD", "score": 87.6, "participants": 150},
    {"name": "Collège Notre-Dame", "city": "Mbanza-Ngungu", "country": "CD", "score": 84.2, "participants": 90},
]

EMI_CHALLENGES = [
    {
        "id": "chal_1",
        "title": "Détecteur de Deepfake 🎥",
        "desc": "Repérez 3 anomalies sur une vidéo ou un audio suspect cette semaine et partagez vos conclusions.",
        "points": 50,
        "participants": 340
    },
    {
        "id": "chal_2",
        "title": "Zéro Partage de Rumeur 🤫",
        "desc": "Ne partagez aucune information non vérifiée pendant 7 jours d'affilée.",
        "points": 100,
        "participants": 620
    },
    {
        "id": "chal_3",
        "title": "Ambassadeur EMI 🗣️",
        "desc": "Aidez un camarade de classe à vérifier une fausse image à l'aide de la recherche d'image inversée.",
        "points": 75,
        "participants": 210
    }
]

# ════════════════════════════════════════════════════════════
# ENDPOINTS EMI — module UNESCO Youth Hackathon 2026
# ════════════════════════════════════════════════════════════
class FactCheckRequest(BaseModel):
    text: str

class SourceSuggestion(BaseModel):
    name:        str
    url:         str
    description: str = ""
    region:      str = ""
    topic:       str = ""
    submitted_by: str = ""

COUNTRY_NAMES = {
    "BI": "Burundi",
    "RW": "Rwanda",
    "CD": "République Démocratique du Congo (RDC)",
    "TZ": "Tanzanie",
    "KE": "Kenya",
    "UG": "Ouganda",
    "SN": "Sénégal",
    "CI": "Côte d'Ivoire",
    "CM": "Cameroun",
    "FR": "France",
}

@app.post("/emi/factcheck/stream")
async def emi_factcheck_stream(req: FactCheckRequest, request: Request):
    text = (req.text or "").strip()
    if not text:
        async def _empty():
            yield f"data: {json.dumps({'type': 'error', 'message': 'Texte vide'})}\n\n"
        return StreamingResponse(_empty(), media_type="text/event-stream")

    country_code = request.headers.get("x-country-code", "BI").upper()
    country_name = COUNTRY_NAMES.get(country_code, "Burundi")
    messages = build_factcheck_messages(text, country_name)
    return StreamingResponse(
        real_stream(messages, max_tokens=800, temperature=0.3, prefer_gemini=False),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

@app.get("/emi/tips")
def emi_tips():
    return {"tips": EMI_TIPS}

@app.get("/emi/sources")
def emi_sources(request: Request, region: Optional[str] = None, topic: Optional[str] = None):
    country_code = request.headers.get("x-country-code", "BI").upper()
    all_sources = EMI_SOURCES + [s for s in EMI_SUGGESTED_SOURCES if s.get("approved")]

    def score(s):
        pts = 0
        # Priorité absolue au pays sélectionné (Burundi, Rwanda, RDC...)
        if country_code in s.get("regions", []):
            pts += 10
            
        # Priorité régionale secondaire (Afrique de l'Est pour BI/RW/TZ/KE/UG, etc.)
        east_africa = ["BI", "RW", "TZ", "KE", "UG"]
        west_africa = ["SN", "CI", "TG", "BJ"]
        central_africa = ["CD", "CM"]
        
        if country_code in east_africa and "afrique_est" in s.get("regions", []):
            pts += 5
        elif country_code in west_africa and "afrique_ouest" in s.get("regions", []):
            pts += 5
        elif country_code in central_africa and "afrique_centrale" in s.get("regions", []):
            pts += 5
            
        if region and region in s.get("regions", []):
            pts += 3
        if topic and topic in s.get("topics", []):
            pts += 3
        if "global" in s.get("regions", []):
            pts += 1
        return pts

    ranked = sorted(all_sources, key=score, reverse=True)
    return {"sources": ranked}

@app.post("/emi/sources/suggest")
def emi_suggest_source(sug: SourceSuggestion):
    """
    Permet à un utilisateur de signaler un organisme de vérification qu'il
    connaît et qui manque à la base (ex: un fact-checkeur local). Stocké en
    mémoire pour cette version — à brancher sur une vraie base de données
    avant une mise en production durable, sinon les suggestions sont
    perdues au redémarrage du serveur.
    """
    if not sug.name.strip() or not sug.url.strip():
        return {"status": "error", "message": "Le nom et l'URL sont obligatoires."}
    entry = {
        "name": sug.name.strip(), "url": sug.url.strip(), "desc": sug.description.strip(),
        "icon": "ti-user-plus", "color": "#7B8BB2",
        "regions": [sug.region] if sug.region else [], "topics": [sug.topic] if sug.topic else [],
        "submitted_by": sug.submitted_by.strip(), "approved": False,
    }
    EMI_SUGGESTED_SOURCES.append(entry)
    print(f"[EMI] Nouvelle suggestion de source reçue : {entry['name']} ({entry['url']})")
    return {"status": "ok", "message": "Merci ! Votre suggestion sera revue avant publication."}

# Pydantic models pour Certificat et Quiz
class CertificateRequest(BaseModel):
    student_name: str
    school_name: Optional[str] = "École Partenaire"
    score_pct: int
    total_questions: int

# Endpoint de génération dynamique de Quiz EMI par Gemini (100% IA, aucune limite, adapté au pays et à la langue)
@app.get("/emi/quiz")
def emi_quiz(request: Request, topic: Optional[str] = "general"):
    lang = request.headers.get("x-language", "fr").lower()
    country_code = request.headers.get("x-country-code", "BI").upper()
    country_name = COUNTRY_NAMES.get(country_code, "Burundi")
    
    lang_names = {"en": "English", "rn": "Kirundi", "sw": "Kiswahili", "fr": "French"}
    target_lang = lang_names.get(lang, "French")

    prompt = f"""Génère un questionnaire d'Éducation aux Médias et à l'Information (EMI) captivant de 5 questions inédites en {target_lang}.
Contexte géographique : {country_name} ({country_code}).
Sujet : {topic} (désinformation, deepfakes, fausses rumeurs santé, arnaques réseaux sociaux, IA générative).

Conserve STRICTEMENT cette structure JSON exacte :
[
  {{
    "q": "Texte de la question ?",
    "opts": ["Option A", "Option B", "Option C", "Option D"],
    "ans": 1,
    "exp": "Explication pédagogique du Coach IA en 1-2 phrases."
  }}
]
Règles :
- "ans" est l'index entier (0, 1, 2 ou 3) de la bonne réponse dans "opts".
- Réponds UNIQUEMENT avec le tableau JSON [...], aucun texte autour, aucune balise markdown.
"""
    try:
        gemini_model = get_gemini_model("gemini-2.0-flash")
        if gemini_model:
            resp = gemini_model.generate_content(prompt).text.strip()
            resp = resp.replace("```json", "").replace("```", "").strip()
            start = resp.find("[")
            end   = resp.rfind("]") + 1
            if start != -1 and end > start:
                quiz_data = json.loads(resp[start:end])
                # Sauvegarder la session de quiz dans Firestore si actif
                if db:
                    try:
                        db.collection("quiz_sessions").add({
                            "country": country_code,
                            "language": lang,
                            "timestamp": firestore.SERVER_TIMESTAMP,
                            "questions_count": len(quiz_data)
                        })
                    except Exception as e:
                        print(f"[FIREBASE] Erreur log session quiz: {e}")
                return {"quiz": quiz_data}
    except Exception as e:
        print(f"[EMI QUIZ IA] Erreur génération Gemini: {e}")

    # Fallback dynamique au cas où l'IA n'est pas joignable
    return {"quiz": EMI_QUIZ}


class SchoolRegistration(BaseModel):
    name: str
    city: str
    country: Optional[str] = None

@app.get("/emi/schools")
def emi_schools(request: Request):
    country_code = request.headers.get("x-country-code", "BI").upper()
    schools = []
    
    # Lecture dynamique depuis Cloud Firestore
    if db:
        try:
            docs = db.collection("schools").stream()
            for doc in docs:
                data = doc.to_dict()
                data["id"] = doc.id
                schools.append(data)
        except Exception as e:
            print(f"[FIREBASE] Erreur lecture des écoles: {e}")

    if not schools:
        schools = EMI_SCHOOLS

    local_schools = [s for s in schools if s.get("country") == country_code]
    other_schools = [s for s in schools if s.get("country") != country_code]
    
    local_schools.sort(key=lambda s: s.get("score", 0), reverse=True)
    other_schools.sort(key=lambda s: s.get("score", 0), reverse=True)
    
    ranked = local_schools + other_schools
    return {
        "user_country": country_code,
        "country_name": COUNTRY_NAMES.get(country_code, country_code),
        "schools": ranked,
        "local_count": len(local_schools)
    }

@app.post("/emi/schools/register")
def emi_register_school(school: SchoolRegistration, request: Request):
    country_code = (school.country or request.headers.get("x-country-code", "BI")).upper()
    if not school.name.strip() or not school.city.strip():
        return {"status": "error", "message": "Le nom de l'école et la ville sont requis."}
    
    new_entry = {
        "name": school.name.strip(),
        "city": school.city.strip(),
        "country": country_code,
        "score": 10.0,
        "participants": 1,
        "created_at": time.time()
    }

    # Sauvegarde dans Cloud Firestore
    if db:
        try:
            doc_ref = db.collection("schools").add(new_entry)
            new_entry["id"] = doc_ref[1].id
        except Exception as e:
            print(f"[FIREBASE] Erreur sauvegarde école Firestore: {e}")

    EMI_SCHOOLS.append(new_entry)
    return {"status": "ok", "message": "Votre école a été enregistrée avec succès !", "school": new_entry}

# Endpoint de génération du Certificat Officiel PDF EMI (ReportLab)
@app.post("/emi/certificate/download")
def emi_download_certificate(req: CertificateRequest, request: Request):
    try:
        from reportlab.lib.pagesizes import A4, landscape
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib import colors

        country_code = request.headers.get("x-country-code", "BI").upper()
        country_name = COUNTRY_NAMES.get(country_code, "Burundi")

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=landscape(A4),
            rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=30
        )
        story = []
        styles = getSampleStyleSheet()

        title_style = ParagraphStyle(
            'TitleStyle',
            parent=styles['Heading1'],
            fontName='Helvetica-Bold',
            fontSize=28,
            leading=34,
            textColor=colors.HexColor('#1A3C8F'),
            alignment=1
        )
        subtitle_style = ParagraphStyle(
            'SubTitleStyle',
            parent=styles['Normal'],
            fontName='Helvetica-Bold',
            fontSize=16,
            leading=20,
            textColor=colors.HexColor('#7C3AED'),
            alignment=1
        )
        body_style = ParagraphStyle(
            'BodyStyle',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=13,
            leading=18,
            textColor=colors.HexColor('#0F1B3D'),
            alignment=1
        )

        story.append(Paragraph("<b>EDUBOT PLATFORM 2026</b>", subtitle_style))
        story.append(Paragraph("Plateforme Numérique Nationale d'Éducation aux Médias et à l'Information", ParagraphStyle('Sub', parent=body_style, fontSize=10, textColor=colors.gray)))
        story.append(Spacer(1, 15))
        story.append(Paragraph("<b>CERTIFICAT D'ACCOMPLISSEMENT</b>", title_style))
        story.append(Spacer(1, 12))
        story.append(Paragraph("Le présent certificat officiel est attribué à :", body_style))
        story.append(Spacer(1, 8))
        story.append(Paragraph(f"<b><font size=22 color='#2251CC'>{req.student_name}</font></b>", body_style))
        story.append(Spacer(1, 8))
        story.append(Paragraph(f"Pour avoir complété avec succès le Parcours d'Éducation aux Médias et à l'Information avec un score de <b>{req.score_pct}%</b>.", body_style))
        story.append(Paragraph(f"Établissement : <b>{req.school_name}</b> | Pays : <b>{country_name}</b>", body_style))
        story.append(Spacer(1, 18))

        # Détection dynamique de l'hôte (serveur/APK backend host)
        host_url = str(request.base_url).rstrip('/')
        cert_id = f"EDUBOT-{country_code}-{int(time.time())}"
        qr_url = f"{host_url}/emi/certificate/verify/{cert_id}"
        
        # Génération du QR Code ReportLab
        qr_drawing = None
        try:
            from reportlab.graphics.shapes import Drawing, Rect
            from reportlab.graphics.barcode import qr
            qr_code = qr.QrCodeWidget(qr_url)
            bounds = qr_code.getBounds()
            width = bounds[2] - bounds[0]
            height = bounds[3] - bounds[1]
            d = Drawing(70, 70, transform=[70.0/width, 0, 0, 70.0/height, 0, 0])
            d.add(qr_code)
            qr_drawing = d
        except Exception as qre:
            print(f"Erreur QR Code: {qre}")

        # Rendu visuel de la Signature Manuscrite / Calligraphiée de la Plateforme
        signature_visual = Paragraph(
            f"<i><font size=18 color='#1A3C8F' name='Times-BoldItalic'>EduBot Platform Authorized</font></i><br/>"
            f"<font size=8 color='#059669'><b>✔ SCELLÉ & VÉRIFIÉ NUMÉRIQUEMENT</b></font><br/>"
            f"<b>EDUBOT AI VERIFIED SIGNATURE</b><br/>"
            f"<font size=7.5 color='#555555'>ID: {cert_id} | Hash: 0x{hash(cert_id) & 0xFFFFFFFF:08X}<br/>"
            f"EDUBOT PLATFORM 2026 — Plateforme Numérique Nationale</font>",
            ParagraphStyle('SigVis', parent=body_style, fontSize=8.5, leading=11, alignment=0)
        )

        qr_cell = qr_drawing if qr_drawing else Paragraph("<b>[QR CODE]</b>", body_style)

        footer_data = [
            [
                qr_cell,
                signature_visual,
                Paragraph(f"<b>Délivré le :</b> {time.strftime('%d/%m/%Y')}<br/><font size=8 color='#17B26A'><b>Statut : OFFICIELLEMENT VALIDÉ</b></font>", body_style)
            ]
        ]
        t = Table(footer_data, colWidths=[90, 430, 200])
        t.setStyle(TableStyle([
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('LINEABOVE', (1,0), (1,0), 0.5, colors.HexColor('#1A3C8F')),
        ]))
        story.append(t)

        doc.build(story)
        pdf_out = buffer.getvalue()
        buffer.close()

        # Enregistrer le certificat dans Firestore
        if db:
            try:
                db.collection("certificates").add({
                    "student_name": req.student_name,
                    "school_name": req.school_name,
                    "country": country_code,
                    "score_pct": req.score_pct,
                    "issued_at": firestore.SERVER_TIMESTAMP
                })
            except Exception as e:
                print(f"[FIREBASE] Erreur sauvegarde certificat: {e}")

        return Response(content=pdf_out, media_type="application/pdf", headers={
            "Content-Disposition": f"attachment; filename=Certificat_EMI_{req.student_name.replace(' ', '_')}.pdf"
        })
    except Exception as e:
        print(f"Erreur PDF: {e}")
        return {"status": "error", "message": f"Erreur de génération PDF: {e}"}

@app.get("/emi/challenges")
def emi_challenges():
    return {"challenges": EMI_CHALLENGES}


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
    """
    Streaming RÉEL, token par token, directement depuis Groq/Gemini.
    Contrairement à /chat (qui passe par le graphe LangGraph complet avec
    relecture qualité + retries — plus robuste mais pas streamable), cet
    endpoint fait un seul appel direct au modèle pour pouvoir renvoyer les
    tokens au fil de l'eau. Le sujet est tout de même classifié pour choisir
    le bon system prompt spécialisé (tech, science, santé, etc.).
    """
    classification = classify_question(req.message)
    lang = classification["language"]
    topic = classification["topic"]

    if topic == "coder":
        # Génération de code : on garde le comportement non-streamé et
        # structuré (plan + code complet), plus adapté à ce cas d'usage.
        loop  = asyncio.get_event_loop()
        start = time.time()
        final = await loop.run_in_executor(None, edubot_graph.invoke, _initial_state(req))
        async def _single_shot():
            meta = {
                "topic": final["topic"], "model_used": final["model_used"],
                "code_plan": final.get("code_plan", []), "language": final.get("language", "fr"),
                "elapsed_ms": int((time.time() - start) * 1000),
            }
            yield f"data: {json.dumps({'type': 'meta', **meta})}\n\n"
            yield f"data: {json.dumps({'type': 'chunk', 'text': final['response']})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        return StreamingResponse(_single_shot(), media_type="text/event-stream",
                                  headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    instruction, prefer_gemini = TOPIC_INSTRUCTIONS.get(topic, TOPIC_INSTRUCTIONS["general"])
    sys_p = get_system_prompt(lang) + "\n\n" + instruction
    messages = build_messages(sys_p, [{"role": h.role, "content": h.content} for h in (req.history or [])], req.message)

    return StreamingResponse(
        real_stream(messages, max_tokens=2048, temperature=0.6, prefer_gemini=prefer_gemini),
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
        "version":   "3.1.0",
        "author":    "Janvier NZAMBIMANA — M1 ITN",
        "topics":    ["tech","science","humanités","santé","général","code"],
        "emi_endpoints": ["/emi/factcheck/stream","/emi/tips","/emi/sources","/emi/quiz"],
        "languages": ["français","anglais","swahili","kirundi"],
        "models":    len(GROQ_MODELS) + len(GEMINI_MODELS),
    }

@app.get("/health")
def health():
    return {"status": "ok", "timestamp": time.time(), "version": "3.1.0"}

print(f"EduBot v3 prêt — {len(GROQ_MODELS) + len(GEMINI_MODELS)} modèles disponibles")
