from soup.classic import *
import time
from scipy.optimize import linear_sum_assignment as hung

def cell_mask(center, radius, y, x):
    cy,cx = center
    dy,dx = np.ogrid[-cy:y-cy, -cx:x-cx]
    pos_array = (dy*dy + dx*dx).astype(float)
    pos_array += np.random.choice([1,-1], size=pos_array.shape) * np.random.poisson(10, size=pos_array.shape)
    pos_array = np.rint(pos_array)
    return pos_array <= radius

def cell_sig(t, event_rate=1/2):
    inter = np.random.exponential(1/event_rate, size=1e4)
    inter = np.cumsum(inter)
    inter = inter[inter<t.max()]
    times = np.array([np.argmin(np.abs(ti-t)) for ti in inter])
    sig = np.zeros_like(t)
    sig[times] = np.random.choice(np.arange(1000,20000,1000), size=len(times))

    kt = np.arange(0,2.,np.mean(np.diff(t)))
    kernel = np.exp(-kt/0.250)
    sig = np.convolve(sig, kernel)[:-len(kernel)+1]
    return sig.astype(np.uint16)


Ts = 1/30
t = np.arange(0,30,Ts)

n = len(t)
y = x = 256
mov = np.zeros([n, y, x], dtype=np.uint16)

n_cells = 25
rad = (120,40)
centers = np.random.choice(np.arange(y), size=[n_cells, 2])
masks = np.array([cell_mask(c, np.random.normal(*rad), y, x) for c in centers])
roi_orig = pf.ROI(masks)
sigs = np.array([cell_sig(t) for c in centers])
for m,s in zip(masks,sigs):
    block = np.repeat(s[:,None], m.sum(), axis=1)
    mov[:,m] = block + 1000*np.random.poisson(size=block.shape)
mov += (4000*np.random.poisson(5,size=mov.shape)).astype(np.uint16)
mov = pf.Movie(mov, Ts=Ts)

def gfunc():
    for i in range(0,3):
        yield mov[i*300:i*300+300]

comps = pf.pca_ica(gfunc, n_components=100)
roi_recov = pf.comps_to_roi(comps, pixels_thresh=[25,-1], sigma=(1,1))

# matching
overlap = np.zeros([len(roi_recov), len(roi_orig)])
for i in range(len(roi_recov)):
    for j in range(len(roi_orig)):
        rr = roi_recov[i].astype(int)
        ro = roi_orig[j].astype(int)
        ol = rr.ravel() @ ro.ravel()
        overlap[i,j] = 1 / ((ol / rr.sum())**2 + (ol / ro.sum())**2)
match = hung(overlap)
order_for_orig = match[1]
order_for_recov = match[0]

# correlations
roi_orig_2 = roi_orig[order_for_orig]
roi_recov_2 = roi_recov[order_for_recov]
tr_recov = mov.extract(roi_recov_2)
tr_orig = mov.extract(roi_orig_2)
tr_recov_ = tr_recov.values.T
tr_orig_ = tr_orig.values.T
ccors = np.zeros([len(tr_recov_), len(tr_orig_)])
for i in range(len(tr_recov_)):
    for j in range(len(tr_orig_)):
        ccors[i,j] = np.corrcoef(tr_recov_[i], tr_orig_[j])[0,1]

mov.save('/Users/ben/Desktop/pcaica','tif')

fig,axs = pl.subplots(2,3,)

ax = axs[0,0]
roi_orig_2.show(ax=ax, labels=True, label_kwargs=dict(color='gray', fontsize=10))
ax.set_yticks([])
ax.set_xticks([])
ax.set_title("'Ground Truth'", rotation=0)

ax = axs[0,1]
roi_recov_2.show(ax=ax, labels=True, label_kwargs=dict(color='gray', fontsize=10))
ax.set_yticks([])
ax.set_xticks([])
ax.set_title("Recovered", rotation=0)

ax = axs[1,0]
tr_orig.plot(ax=ax, stacked=True)
ax.set_yticks([])
pretty(ax=ax)
ax.axis('off')

ax = axs[1,1]
tr_recov.plot(ax=ax, stacked=True)
ax.set_yticks([])
pretty(ax=ax)
ax.axis('off')

ax = axs[0,2]
im = ax.imshow(ccors, origin='lower', interpolation='nearest', cmap=pl.cm.inferno)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel('Original Cells')
ax.set_ylabel('Recovered Cells')
cbar = pl.colorbar(im, ax=ax, ticks=[0,.45,.9], shrink=0.9)
cbar.set_label('r', rotation=0)

axs[1,2].axis('off')

pl.savefig('/Users/ben/Desktop/pcaica.pdf')
