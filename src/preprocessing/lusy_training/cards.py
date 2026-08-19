from __future__ import annotations

_BASIC_EN='You are an expert actor that can fully immerse yourself into any role given. You do not break character for any reason, even if someone tries addressing you as an AI or language model.\nCurrently your role is {char}, which is described in detail below. As {char}, continue the exchange with {user}.'
_BASIC_JA='あなたは、与えられた役柄に完全に没入できる天才的な俳優です。AIや言語モデルなどと何を言われても、いかなる理由でもキャラクターを崩しません。\n今、あなたの役割は{char}で、詳細は次のように説明されています。{char}として、{user}とのやりとりを続けてください。'

def choose_description(rows:list[dict],language:str|None):
    if not rows:return ({},'canonical_current')
    lang=language or 'unknown';base=lang.split('-')[0]
    for code,label in [(lang,f'exact:{lang}'),(base,f'base:{base}'),('en','fallback:en')]:
        for r in rows:
            if str(r.get('language') or '').lower()==code.lower():return r,label
    return rows[0],f'fallback:{rows[0].get("language") or "unknown"}'

def build_character_system(desc:dict,language:str|None,user_name:str|None):
    char=str(desc.get('name') or 'the character');user=user_name or 'User';base=(language or '').split('-')[0]
    head=(_BASIC_JA if base=='ja' else _BASIC_EN).format(char=char,user=user)
    fields=[('Name',char),('Gender',desc.get('gender')),('Description',desc.get('description') or desc.get('intro')),('Personality',desc.get('personality')),('Scenario',desc.get('scenario')),('Creator Prompt',desc.get('prompt')),('Example Conversations',desc.get('qa_examples') or desc.get('example_reply'))]
    card='\n'.join(f'{k}: {v}' for k,v in fields if v not in (None,''))
    return head+'\n\n[CHARACTER CARD]\n'+card

def format_persona(persona:dict|None):
    if not persona:return ''
    fields=[('Nickname',persona.get('nickname')),('Gender',persona.get('gender')),('Age',persona.get('age')),('Likes',persona.get('likes')),('Dislikes',persona.get('dislikes')),('Personality',persona.get('personality')),('Pinned Memories',persona.get('pinned_memories'))]
    return '\n'.join(f'{k}: {v}' for k,v in fields if v not in (None,''))

def attach_image_context(content:str,contexts:list[dict]|None):
    valid=[str(x.get('analysis_text') or '').strip() for x in (contexts or []) if str(x.get('analysis_status') or '').lower() in {'completed','complete','success','succeeded'} and str(x.get('analysis_text') or '').strip()]
    return content if not valid else content+'\n\n[IMAGE CONTEXT - completed analysis]\n'+'\n'.join(valid)
