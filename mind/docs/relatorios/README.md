# Relatórios de implementação

Um relatório em PDF por sessão de trabalho concluída. Cada um cobre:

1. **Sumário executivo** — o que foi entregue
2. **O que fiz** — componentes e ficheiros
3. **Como fiz** — método, verificações executadas e decisões técnicas
4. **O que não fiz e porquê** — separando o que ficou fora de âmbito por
   indicação da especificação das lacunas reais da implementação
5. **O que poderia fazer** — possibilidades que a arquitectura suporta
6. **O que vou fazer a seguir** — plano ordenado por prioridade

Nomenclatura: `Relatorio_MIND_AAAA-MM-DD[_NN_assunto].pdf` — o sufixo
distingue vários relatórios do mesmo dia.

| Relatório | Âmbito |
|---|---|
| `Relatorio_MIND_2026-08-03.pdf` | Base do MIND (commit `ac25c16`) e extensão HIPPOCAMPUS (commit `fdcb706`) |
| `Relatorio_MIND_2026-08-03_02_suite_de_testes.pdf` | Suite de testes — ponto 1 do plano (commit `4f51da6`) |
| `Relatorio_MIND_2026-08-03_03_fecho_de_lacunas.pdf` | Pontos 2 a 6: diff de contrato, limpeza git, decisão LangGraph, esquema JSON (commit `7c6d807`) |
| `Relatorio_MIND_2026-08-03_04_huggingface_gpu_alugada.pdf` | Router para HuggingFace em GPU alugada e avaliação sobre modelos retreinados (commit `deec79d`) |
