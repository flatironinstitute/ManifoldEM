
class EigValCanvas(FigureCanvas):
    # all eigenvecs/vals:
    eig_n = []
    eig_v = []
    # user-computed vecs/vals (color blue):
    eig_n1 = []
    eig_v1 = []
    # remaining vecs/vals via [eig_n - eig_n1] (color gray):
    eig_n2 = []
    eig_v2 = []

    def __init__(self, parent=None, width=5, height=4, dpi=200):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        self.axes.clear()
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)
        self.updateGeometry()
        fig.set_tight_layout(True)
        self.plot()

    def EigValRead(self):

        EigValCanvas.eig_n = []
        EigValCanvas.eig_v = []
        EigValCanvas.eig_n1 = []
        EigValCanvas.eig_v1 = []
        EigValCanvas.eig_n2 = []
        EigValCanvas.eig_v2 = []

        fname = os.path.join(p.out_dir, 'topos', f'PrD_{P4.user_PrD}', 'eig_spec.txt')
        data = []
        with open(fname) as values:
            for column in zip(*[line for line in csv.reader(values, dialect="excel-tab")]):
                data.append(column)
        col1 = data[0]
        col2 = data[1]
        cols = np.column_stack((col1, col2))

        for i, j in cols:
            EigValCanvas.eig_n.append(int(i))
            EigValCanvas.eig_v.append(float(j))
            if int(i) <= int(p.num_psis):
                EigValCanvas.eig_n1.append(int(i))
                EigValCanvas.eig_v1.append(float(j))
            else:
                EigValCanvas.eig_n2.append(int(i))
                EigValCanvas.eig_v2.append(float(j))
        return

    def plot(self):
        ax = self.figure.add_subplot(111)
        ax.clear()
        ax.bar(EigValCanvas.eig_n1, EigValCanvas.eig_v1, edgecolor='none', color='#1f77b4', align='center')  #C0: blue
        ax.bar(EigValCanvas.eig_n2, EigValCanvas.eig_v2, edgecolor='none', color='#7f7f7f', align='center')  #C7: gray

        for tick in ax.xaxis.get_major_ticks():
            tick.label1.set_fontsize(6)
        for tick in ax.yaxis.get_major_ticks():
            tick.label1.set_fontsize(6)
        ax.set_title('Eigenvalue Spectrum', fontsize=8)
        ax.set_xlabel(r'$\mathrm{\Psi}$', fontsize=8)
        ax.set_ylabel(r'$\mathrm{\lambda}$', fontsize=8, rotation=0)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.axhline(0, color='k', linestyle='-', linewidth=.25)
        ax.get_xaxis().set_tick_params(direction='out', width=.25, length=2)
        ax.get_yaxis().set_tick_params(direction='out', width=.25, length=2)
        ax.grid(linestyle='-', linewidth='0.5', color='lightgray', alpha=0.2)
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        ax.yaxis.offsetText.set_fontsize(6)
        ax.set_xticks(EigValCanvas.eig_n)
        ax.autoscale()
        self.draw()

