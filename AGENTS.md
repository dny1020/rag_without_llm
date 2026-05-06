## Creacion RAG

---
### 1. comportamiento sin llm
    - la fuente esta en docs/wiki/
    - se debe crear con python y fastapi
    - PostgreSQL + pgvector
    - streamlit frontend
    - debe tener memoria en cada conversacion
    - se despliega local por ahora en contenedor para ingresar por localhost

### 2. diseño

```ascii
conversations
 ├── id
 ├── user_id
 └── created_at

messages
 ├── id
 ├── conversation_id
 ├── role (user/system/assistant)
 ├── content
 └── created_at
```

### 3. librerias recomendadas


```ascii
Backend
FastAPI
SQLAlchemy
Base de datos
PostgreSQL
Opcional (rendimiento)
Redis → cache / sesiones
```
### 4. estuctura

```bash
.
├── main.py
├── requirements.txt
├── .env
├── logs/
│   └── app.log
├── data/
│   └── .gitkeep
└── src/
    ├── __init__.py
    ├── api/
    │   ├── __init__.py
    │   └── routes.py
    ├── core/
    │   ├── __init__.py
    │   └── config.py
    ├── db/
    │   ├── __init__.py
    │   ├── session.py
    │   └── models.py
    ├── schemas/
    │   ├── __init__.py
    │   └── message.py
    ├── services/
    │   ├── __init__.py
    │   ├── rag.py
    │   └── chat.py
    └── helpers/
        ├── __init__.py
        ├── logger.py
        └── utils.py
```