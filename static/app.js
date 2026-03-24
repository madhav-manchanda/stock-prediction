// ===== Global State =====
let priceChart = null;

// ===== Event Listeners =====
document.getElementById('tickerInput').addEventListener('keydown', (e) => {
    if (e.key === 'Enter') runPrediction();
});

// ===== Slider =====
function updateSliderLabel() {
    const val = document.getElementById('daysSlider').value;
    document.getElementById('daysValue').textContent = `${val} days`;
}

// ===== Quick Select =====
function selectTicker(ticker) {
    document.getElementById('tickerInput').value = ticker;
    runPrediction();
}

// ===== Main Prediction Flow =====
async function runPrediction() {
    const ticker = document.getElementById('tickerInput').value.trim().toUpperCase();
    const days = document.getElementById('daysSlider').value;

    if (!ticker) {
        showError('Please enter a stock ticker symbol');
        return;
    }

    showLoading(true);
    hideError();

    try {
        // Fetch prediction and stock info in parallel
        const [predRes, infoRes] = await Promise.all([
            fetch(`/api/predict?ticker=${encodeURIComponent(ticker)}&days=${days}`),
            fetch(`/api/stock-info?ticker=${encodeURIComponent(ticker)}`)
        ]);

        const predData = await predRes.json();
        const infoData = await infoRes.json();

        if (predData.error) throw new Error(predData.error);
        if (infoData.error) throw new Error(infoData.error);

        // Update UI
        updateStockInfo(infoData);
        updatePredictionSummary(predData);
        renderChart(predData);

        // Show results
        document.getElementById('emptyState').style.display = 'none';
        document.getElementById('resultsSection').classList.add('active');

    } catch (err) {
        showError(err.message || 'Something went wrong');
    } finally {
        showLoading(false);
    }
}

// ===== Update Stock Info =====
function updateStockInfo(info) {
    document.getElementById('stockSymbol').textContent = info.symbol;
    document.getElementById('stockName').textContent = info.name;
    document.getElementById('currentPrice').textContent = formatCurrency(info.currentPrice, info.currency);

    const changeEl = document.getElementById('priceChange');
    const isPositive = info.change >= 0;
    changeEl.className = `price-change ${isPositive ? 'positive' : 'negative'}`;
    changeEl.textContent = `${isPositive ? '▲' : '▼'} ${Math.abs(info.change).toFixed(2)} (${Math.abs(info.changePercent).toFixed(2)}%)`;

    document.getElementById('highValue').textContent = formatCurrency(info.fiftyTwoWeekHigh, info.currency);
    document.getElementById('lowValue').textContent = formatCurrency(info.fiftyTwoWeekLow, info.currency);
    document.getElementById('volumeValue').textContent = formatNumber(info.volume);
    document.getElementById('sectorValue').textContent = info.sector || 'N/A';

    document.getElementById('chartTicker').textContent = `${info.symbol} • ${info.name}`;
}

// ===== Update Prediction Summary =====
function updatePredictionSummary(data) {
    const days = document.getElementById('daysSlider').value;
    document.getElementById('predLabel').textContent = `Predicted in ${days} days`;

    const predPriceEl = document.getElementById('predictedPrice');
    const isPositive = data.priceChange >= 0;
    predPriceEl.textContent = `$${data.predictedEndPrice.toFixed(2)}`;
    predPriceEl.className = `predicted-price ${isPositive ? 'positive' : 'negative'}`;

    const changeEl = document.getElementById('predictionChange');
    changeEl.textContent = `${isPositive ? '▲' : '▼'} $${Math.abs(data.priceChange).toFixed(2)} (${Math.abs(data.priceChangePercent).toFixed(2)}%)`;
    changeEl.className = `prediction-change ${isPositive ? 'positive' : 'negative'}`;

    document.getElementById('accuracyValue').textContent = `${data.metrics.accuracy.toFixed(1)}%`;
    document.getElementById('mapeValue').textContent = `${data.metrics.mape.toFixed(2)}%`;
    document.getElementById('trainLossValue').textContent = data.metrics.trainLoss.toFixed(5);
    document.getElementById('valLossValue').textContent = data.metrics.valLoss.toFixed(5);
}

// ===== Render Chart =====
function renderChart(data) {
    const ctx = document.getElementById('priceChart').getContext('2d');

    if (priceChart) priceChart.destroy();

    const historicalDates = data.historical.dates;
    const historicalPrices = data.historical.prices;
    const predictionDates = data.predictions.dates;
    const predictionPrices = data.predictions.prices;

    // Connector: last historical point + first prediction
    const connectorDates = [historicalDates[historicalDates.length - 1], predictionDates[0]];
    const connectorPrices = [historicalPrices[historicalPrices.length - 1], predictionPrices[0]];

    const isPositive = data.priceChange >= 0;

    priceChart = new Chart(ctx, {
        type: 'line',
        data: {
            datasets: [
                {
                    label: 'Historical Price',
                    data: historicalDates.map((d, i) => ({ x: d, y: historicalPrices[i] })),
                    borderColor: '#6366f1',
                    backgroundColor: createGradient(ctx, '#6366f1', 0.15),
                    borderWidth: 2,
                    fill: true,
                    tension: 0.4,
                    pointRadius: 0,
                    pointHoverRadius: 5,
                    pointHoverBackgroundColor: '#6366f1',
                },
                {
                    label: 'Connector',
                    data: connectorDates.map((d, i) => ({ x: d, y: connectorPrices[i] })),
                    borderColor: 'rgba(255,255,255,0.2)',
                    borderWidth: 2,
                    borderDash: [6, 4],
                    fill: false,
                    tension: 0,
                    pointRadius: 0,
                },
                {
                    label: 'Predicted Price',
                    data: predictionDates.map((d, i) => ({ x: d, y: predictionPrices[i] })),
                    borderColor: isPositive ? '#10b981' : '#ef4444',
                    backgroundColor: createGradient(ctx, isPositive ? '#10b981' : '#ef4444', 0.1),
                    borderWidth: 2,
                    fill: true,
                    tension: 0.4,
                    pointRadius: 0,
                    pointHoverRadius: 5,
                    pointHoverBackgroundColor: isPositive ? '#10b981' : '#ef4444',
                    borderDash: [6, 3],
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',
                intersect: false,
            },
            plugins: {
                legend: {
                    display: true,
                    position: 'top',
                    labels: {
                        color: '#94a3b8',
                        font: { family: 'Inter', size: 12 },
                        usePointStyle: true,
                        pointStyle: 'circle',
                        padding: 20,
                        filter: (item) => item.text !== 'Connector',
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(17, 24, 39, 0.95)',
                    titleColor: '#f1f5f9',
                    bodyColor: '#94a3b8',
                    borderColor: 'rgba(255,255,255,0.1)',
                    borderWidth: 1,
                    cornerRadius: 8,
                    padding: 12,
                    titleFont: { family: 'Inter', weight: '600' },
                    bodyFont: { family: 'Inter' },
                    callbacks: {
                        label: (ctx) => {
                            if (ctx.dataset.label === 'Connector') return null;
                            return `${ctx.dataset.label}: $${ctx.parsed.y.toFixed(2)}`;
                        }
                    }
                },
            },
            scales: {
                x: {
                    type: 'time',
                    time: {
                        unit: 'month',
                        displayFormats: { month: 'MMM yyyy' }
                    },
                    grid: {
                        color: 'rgba(255,255,255,0.04)',
                        drawBorder: false,
                    },
                    ticks: {
                        color: '#64748b',
                        font: { family: 'Inter', size: 11 },
                        maxTicksLimit: 8,
                    }
                },
                y: {
                    grid: {
                        color: 'rgba(255,255,255,0.04)',
                        drawBorder: false,
                    },
                    ticks: {
                        color: '#64748b',
                        font: { family: 'Inter', size: 11 },
                        callback: (val) => '$' + val.toFixed(0),
                    }
                }
            }
        }
    });
}

function createGradient(ctx, color, alpha) {
    const gradient = ctx.createLinearGradient(0, 0, 0, 350);
    gradient.addColorStop(0, hexToRgba(color, alpha));
    gradient.addColorStop(1, hexToRgba(color, 0));
    return gradient;
}

// ===== Utilities =====
function formatCurrency(value, currency = 'USD') {
    if (!value) return '—';
    const symbol = currency === 'INR' ? '₹' : '$';
    return `${symbol}${Number(value).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
}

function formatNumber(num) {
    if (!num) return '—';
    if (num >= 1e9) return (num / 1e9).toFixed(1) + 'B';
    if (num >= 1e6) return (num / 1e6).toFixed(1) + 'M';
    if (num >= 1e3) return (num / 1e3).toFixed(1) + 'K';
    return num.toLocaleString();
}

function hexToRgba(hex, alpha) {
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);
    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

// ===== Loading =====
function showLoading(show) {
    const overlay = document.getElementById('loadingOverlay');
    const btn = document.getElementById('predictBtn');

    if (show) {
        overlay.classList.add('active');
        btn.disabled = true;
    } else {
        overlay.classList.remove('active');
        btn.disabled = false;
    }
}

// ===== Error =====
function showError(msg) {
    const toast = document.getElementById('errorToast');
    toast.textContent = `❌ ${msg}`;
    toast.classList.add('active');
    setTimeout(() => toast.classList.remove('active'), 5000);
}

function hideError() {
    document.getElementById('errorToast').classList.remove('active');
}
