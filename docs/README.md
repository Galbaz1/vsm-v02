# Documentation Index

**Last Updated:** 2025-11-25

This directory contains all technical documentation for VSM Demo v02.

---

## 📚 Main Documents

### [ARCHITECTURE.md](ARCHITECTURE.md)
**The authoritative system design document**

Comprehensive overview covering:
- ✅ Dual RAG pipeline architecture (text + multimodal)
- ✅ Technology stack and component interactions
- ✅ Data flows (ingestion and search)
- ✅ API endpoints and schemas
- ✅ Deployment workflow
- ✅ Troubleshooting guide

**Audience:** Developers, system architects  
**Status:** ✅ Current (2025-11-25)

---

### [RAG_PIPELINE_EXPLAINED.md](RAG_PIPELINE_EXPLAINED.md)
**Deep-dive into the regular text-based RAG pipeline**

Covers:
- LandingAI ADE parsing process
- Weaviate schema design for `AssetManual` collection
- Ollama embedding generation with `nomic-embed-text`
- Vector search implementation
- Bounding box and visual grounding metadata

**Audience:** Backend developers  
**Status:** ✅ Current (2025-11-25)

---

### [COLQWEN_INGESTION_EXPLAINED.md](COLQWEN_INGESTION_EXPLAINED.md)
**Deep-dive into ColQwen multimodal RAG pipeline**

Covers:
- ColQwen2.5 model architecture
- Multi-vector embedding generation
- Late interaction (MaxSim) search
- Weaviate multi-vector schema
- Memory and performance considerations
- Comparison with regular RAG

**Audience:** ML engineers, backend developers  
**Status:** ✅ Current (2025-11-25)

---

## 🧪 Testing & Deployment

### [../TESTING.md](../TESTING.md)
**Testing procedures and verification**

Covers:
- Component testing (fast vector, ColQwen, agent)
- API endpoint testing with curl examples
- Verification scripts
- Known limitations
- Troubleshooting common issues

**Audience:** QA engineers, developers  
**Status:** ✅ Current (2025-11-25)

---

## 📖 Reference Guides

### [USING_WITH_ELYSIA.md](USING_WITH_ELYSIA.md)
**Optional integration with Weaviate's Elysia framework**

Covers:
- What is Elysia and when to use it
- Integration options (standalone vs. hybrid)
- Setup instructions for local Weaviate
- Pros/cons compared to custom FastAPI

**Audience:** Developers considering alternatives  
**Status:** ⚠️ Reference only (Elysia not used in current implementation)

---

## 🗂️ Archived / Legacy

The following files contain historical information but may be outdated:

### ../project.md
Early project context for Cursor AI. Some information superseded by ARCHITECTURE.md.

**Status:** ⚠️ Partially outdated (mentions Elysia, bge-m3 instead of nomic-embed-text)

### ../todo.md
Scratchpad notes on ColQwen integration planning.

**Status:** ⚠️ Outdated (work completed)

### ../colqwen/
Standalone ColQwen FastAPI service (port 8002).

**Status:** ⚠️ Deprecated (integrated into main API at `api/services/colqwen.py`)

---

## 📌 Quick Reference

| Need to... | Read this |
|-----------|-----------|
| Understand the overall system | [ARCHITECTURE.md](ARCHITECTURE.md) |
| Learn how text search works | [RAG_PIPELINE_EXPLAINED.md](RAG_PIPELINE_EXPLAINED.md) |
| Learn how visual search works | [COLQWEN_INGESTION_EXPLAINED.md](COLQWEN_INGESTION_EXPLAINED.md) |
| Test the system | [../TESTING.md](../TESTING.md) |
| Deploy to production | [ARCHITECTURE.md#deployment-workflow](ARCHITECTURE.md#deployment-workflow) |
| Troubleshoot issues | [ARCHITECTURE.md#troubleshooting](ARCHITECTURE.md#troubleshooting) |
| Use ingestion scripts | [../scripts/README.md](../scripts/README.md) |
| Develop frontend | [../frontend/README.md](../frontend/README.md) |

---

## 🔄 Document Maintenance

### Keeping Docs Current

When making code changes, update these docs:

| Code Change | Update Document |
|------------|----------------|
| New API endpoint | ARCHITECTURE.md (API Endpoints section) |
| Schema change | ARCHITECTURE.md + RAG_PIPELINE_EXPLAINED.md |
| New script | ARCHITECTURE.md (Scripts Reference) |
| Deployment change | ARCHITECTURE.md (Deployment Workflow) |
| New dependency | ARCHITECTURE.md (Technology Stack) |

### Version History

- **v1.0** (2025-11-25): Initial consolidated documentation
  - Created ARCHITECTURE.md
  - Updated README.md
  - Created this index

---

## 📞 Support

For questions:
1. Check ARCHITECTURE.md first
2. Review specific technical deep-dives
3. Consult inline code comments
4. Run verification scripts in TESTING.md
