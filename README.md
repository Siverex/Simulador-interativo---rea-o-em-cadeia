# Simulador interativo---reação em cadeia
Esse é um programa simples de python que faz um gráfico interativo de acordo com a sua escolha de numero de fissões  

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


def gerar_atomos(n_atomos, tamanho_area):
    return np.random.uniform(0, tamanho_area, size=(n_atomos, 2))


def simular_reacao_cadeia(n_atomos, tamanho_area, k, raio_captura, max_geracoes, neutrons_iniciais=2):
    posicoes_atomos = gerar_atomos(n_atomos, tamanho_area)
    fissionado = np.zeros(n_atomos, dtype=bool)

    neutrons_iniciais = min(neutrons_iniciais, n_atomos)
    indices_iniciais = np.random.choice(n_atomos, size=neutrons_iniciais, replace=False)
    neutrons_ativos = posicoes_atomos[indices_iniciais].copy()

    fissoes_por_geracao = [0]

    for geracao in range(max_geracoes):
        if len(neutrons_ativos) == 0:
            break

        novos_neutrons = []
        fissoes_nesta_geracao = 0

        for neutron in neutrons_ativos:
            distancias = np.linalg.norm(posicoes_atomos - neutron, axis=1)
            candidatos = np.where((distancias < raio_captura) & (~fissionado))[0]

            if len(candidatos) > 0:
                alvo = candidatos[np.argmin(distancias[candidatos])]
                fissionado[alvo] = True
                fissoes_nesta_geracao += 1

                n_emitidos = np.random.poisson(k)
                for _ in range(n_emitidos):
                    angulo = np.random.uniform(0, 2 * np.pi)
                    passo = np.random.uniform(raio_captura * 2, raio_captura * 5)
                    nova_pos = posicoes_atomos[alvo] + passo * np.array([np.cos(angulo), np.sin(angulo)])
                    nova_pos = np.clip(nova_pos, 0, tamanho_area)
                    novos_neutrons.append(nova_pos)

        fissoes_por_geracao.append(fissoes_nesta_geracao)
        neutrons_ativos = np.array(novos_neutrons) if novos_neutrons else np.empty((0, 2))

    return posicoes_atomos, fissionado, fissoes_por_geracao


class AppReacaoCadeia:
    def __init__(self, root):
        self.root = root
        self.root.title("Simulador Interativo de Reação em Cadeia")
        self.root.geometry("1000x680")

        painel_controles = ttk.Frame(root)
        painel_controles.pack(side="left", fill="y", padx=10, pady=10)

        painel_graficos = ttk.Frame(root)
        painel_graficos.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        ttk.Label(painel_controles, text="Parâmetros da simulação", font=("Segoe UI", 11, "bold")).pack(
            anchor="w", pady=(0, 10))

        ttk.Label(painel_controles, text="Número de núcleos:").pack(anchor="w")
        self.e_n_atomos = ttk.Entry(painel_controles)
        self.e_n_atomos.insert(0, "800")
        self.e_n_atomos.pack(fill="x", pady=(0, 10))

        ttk.Label(painel_controles, text="Nêutrons iniciais:").pack(anchor="w")
        self.e_neutrons_iniciais = ttk.Entry(painel_controles)
        self.e_neutrons_iniciais.insert(0, "2")
        self.e_neutrons_iniciais.pack(fill="x", pady=(0, 10))

        ttk.Label(painel_controles, text="Número de gerações:").pack(anchor="w")
        self.var_geracoes = tk.IntVar(value=10)
        self.slider_geracoes = ttk.Scale(painel_controles, from_=1, to=25, orient="horizontal",
                                         variable=self.var_geracoes, command=self._atualizar_label_geracoes)
        self.slider_geracoes.pack(fill="x")
        self.label_geracoes = ttk.Label(painel_controles, text="10 gerações")
        self.label_geracoes.pack(anchor="w", pady=(0, 10))

        ttk.Label(painel_controles, text="Fator de multiplicação k:").pack(anchor="w")
        self.var_k = tk.DoubleVar(value=1.5)
        self.slider_k = ttk.Scale(painel_controles, from_=0.3, to=3.0, orient="horizontal",
                                  variable=self.var_k, command=self._atualizar_label_k)
        self.slider_k.pack(fill="x")
        self.label_k = ttk.Label(painel_controles, text="k = 1.50 (supercrítico)")
        self.label_k.pack(anchor="w", pady=(0, 10))

        ttk.Label(painel_controles, text="Raio de captura:").pack(anchor="w")
        self.e_raio = ttk.Entry(painel_controles)
        self.e_raio.insert(0, "4.0")
        self.e_raio.pack(fill="x", pady=(0, 15))

        ttk.Button(painel_controles, text="Simular", command=self._on_simular).pack(fill="x", pady=(0, 5))
        ttk.Button(painel_controles, text="Nova semente aleatória", command=self._nova_semente).pack(fill="x")

        self.resultado_texto = tk.Text(painel_controles, height=8, width=32, state="disabled", bg="#f5f5f5")
        self.resultado_texto.pack(fill="x", pady=(15, 0))

        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.ax_mapa = self.fig.add_subplot(2, 2, (1, 3))
        self.ax_barras = self.fig.add_subplot(2, 2, 2)
        self.ax_acumulado = self.fig.add_subplot(2, 2, 4)
        self.fig.tight_layout(pad=3)

        self.canvas = FigureCanvasTkAgg(self.fig, master=painel_graficos)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.canvas.draw()

        self._nova_semente()
        self._on_simular()

    def _atualizar_label_geracoes(self, _evento=None):
        self.label_geracoes.config(text=f"{int(self.var_geracoes.get())} gerações")

    def _atualizar_label_k(self, _evento=None):
        k = self.var_k.get()
        if k < 1.0:
            categoria = "subcrítico"
        elif k == 1.0:
            categoria = "crítico"
        else:
            categoria = "supercrítico"
        self.label_k.config(text=f"k = {k:.2f} ({categoria})")

    def _nova_semente(self):
        self.semente = np.random.randint(0, 1_000_000)

    def _on_simular(self):
        try:
            n_atomos = int(self.e_n_atomos.get())
            neutrons_iniciais = int(self.e_neutrons_iniciais.get())
            max_geracoes = int(self.var_geracoes.get())
            k = float(self.var_k.get())
            raio_captura = float(self.e_raio.get())

            if n_atomos <= 0 or neutrons_iniciais <= 0 or raio_captura <= 0:
                raise ValueError("Todos os valores devem ser positivos.")
            if neutrons_iniciais > n_atomos:
                raise ValueError("Nêutrons iniciais não pode exceder o número de núcleos.")

            np.random.seed(self.semente)
            tamanho_area = 100
            posicoes_atomos, fissionado, fissoes_por_geracao = simular_reacao_cadeia(
                n_atomos, tamanho_area, k, raio_captura, max_geracoes, neutrons_iniciais
            )

            self._desenhar(posicoes_atomos, fissionado, fissoes_por_geracao, tamanho_area, k)

        except ValueError as e:
            messagebox.showerror("Erro de entrada", f"Verifique os valores digitados.\n{e}")

    def _desenhar(self, posicoes_atomos, fissionado, fissoes_por_geracao, tamanho_area, k):
        self.ax_mapa.clear()
        self.ax_barras.clear()
        self.ax_acumulado.clear()

        self.ax_mapa.scatter(posicoes_atomos[~fissionado, 0], posicoes_atomos[~fissionado, 1],
                             c="tab:blue", s=18, label="Intacto")
        self.ax_mapa.scatter(posicoes_atomos[fissionado, 0], posicoes_atomos[fissionado, 1],
                             c="tab:red", s=18, label="Fissionado")
        self.ax_mapa.set_xlim(0, tamanho_area)
        self.ax_mapa.set_ylim(0, tamanho_area)
        self.ax_mapa.set_title("Estado final dos núcleos")
        self.ax_mapa.legend(loc="upper right", fontsize=8)
        self.ax_mapa.set_xticks([])
        self.ax_mapa.set_yticks([])

        geracoes = np.arange(len(fissoes_por_geracao))
        self.ax_barras.bar(geracoes, fissoes_por_geracao, color="tab:orange")
        self.ax_barras.set_title("Fissões por geração")
        self.ax_barras.set_xlabel("Geração")
        self.ax_barras.set_ylabel("Fissões")

        acumulado = np.cumsum(fissoes_por_geracao)
        self.ax_acumulado.plot(geracoes, acumulado, marker="o", color="tab:red")
        self.ax_acumulado.set_title("Fissões acumuladas")
        self.ax_acumulado.set_xlabel("Geração")
        self.ax_acumulado.set_ylabel("Total")

        self.fig.tight_layout(pad=3)
        self.canvas.draw()

        total_fissoes = int(fissionado.sum())
        n_atomos = len(posicoes_atomos)
        texto = (
            f"k = {k:.2f}\n"
            f"Núcleos totais: {n_atomos}\n"
            f"Núcleos fissionados: {total_fissoes}\n"
            f"Fração fissionada: {100 * total_fissoes / n_atomos:.1f}%\n"
            f"Gerações efetivas: {len(fissoes_por_geracao) - 1}"
        )
        self.resultado_texto.config(state="normal")
        self.resultado_texto.delete("1.0", tk.END)
        self.resultado_texto.insert(tk.END, texto)
        self.resultado_texto.config(state="disabled")


if __name__ == "__main__":
    root = tk.Tk()
    app = AppReacaoCadeia(root)
    root.mainloop()
