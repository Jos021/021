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
| `Relatorio_MIND_2026-08-03_05_sandbox_evolutiva.pdf` | Sandbox de testes evolutiva — instruções 45-61 (commit `06a57a6`) |
| `Relatorio_MIND_2026-08-03_06_preparacao_do_piloto.pdf` | Ferramenta de verificação e medição com modelos reais (commit `a143abf`) |
| `Relatorio_MIND_2026-08-03_07_ensaio_do_runbook.pdf` | Ensaio das 4 passagens sem GPU; discrepância do `ml-status` corrigida (commit `de9e0a1`) |
| `Relatorio_MIND_2026-08-03_08_consolidado.pdf` | Consolidado até 3 de Agosto — substituído pelo 11 |
| `Relatorio_MIND_2026-08-05_09_modo_anthropic.pdf` | Quinto modo do router: API da Anthropic directa (commit `a71e5f8`) |
| `Relatorio_MIND_2026-08-05_10_o_que_esta_provado.pdf` | Ensaio do modo Anthropic; defeito de medição da conformidade corrigido; matriz do que está e não está provado (commit `43068f2`) |
| **`Relatorio_MIND_2026-08-05_11_consolidado.pdf`** | **PONTO DE ENTRADA — estado completo: 13 entregas, 8 defeitos corrigidos, o que está e não está provado (commit `8d58276`)** |

O **relatório 11 (consolidado)** é o ponto de entrada para quem chega ao
projecto: resume as 13 entregas, as decisões de arquitectura, os defeitos
reais encontrados, e — sobretudo — separa linha a linha o que está provado
do que continua por provar. Substitui o consolidado 08, que é anterior ao
modo Anthropic.
