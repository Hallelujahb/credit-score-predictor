import json
from pathlib import Path

from django.http import JsonResponse
from django.shortcuts import render, redirect

from prediction.ml.predict import score_borrower
from .forms import CreditApplicantForm, DatasetUploadForm
from .ml_model import CreditPredictor
from .models import CreditApplicant, CreditApplication, SystemStatus, TrainingDataset, TrainingRun

APP_DIR = Path(__file__).resolve().parent
LINEAR_MODEL_PATH = APP_DIR / 'linear_predictor.joblib'
CATBOOST_MODEL_PATH = APP_DIR / 'catboost_predictor.cbm'


def predict_price(request):
    prediction = None
    error = None

    if request.method == 'POST':
        try:
            monthly_income = float(request.POST.get('monthly_income'))
            avg_utility_bill = float(request.POST.get('avg_utility_bill'))
            mobile_spend = float(request.POST.get('mobile_spend'))
            savings_amount = float(request.POST.get('savings_amount'))
            years_at_job = int(request.POST.get('years_at_job'))
            utilization_rate = float(request.POST.get('utilization_rate'))
            days_since_overdraft = int(request.POST.get('days_since_overdraft'))
            is_entrepreneur = int(request.POST.get('is_entrepreneur'))
            past_on_time_payments = int(request.POST.get('past_on_time_payments'))

            predictor = CreditPredictor()
            prediction = predictor.predict(
                monthly_income,
                avg_utility_bill,
                mobile_spend,
                savings_amount,
                years_at_job,
                utilization_rate,
                days_since_overdraft,
                is_entrepreneur,
                past_on_time_payments,
            )
        except (ValueError, TypeError):
            error = 'Invalid input values.'
        except FileNotFoundError as exc:
            error = str(exc)

    return render(request, 'predict.html', {'prediction': prediction, 'error': error})


def parse_request_data(request):
    if request.content_type == 'application/json':
        try:
            return json.loads(request.body.decode('utf-8') or '{}')
        except json.JSONDecodeError:
            return {}
    return request.POST if request.method == 'POST' else request.GET


def predict_api(request):
    data = parse_request_data(request)
    model_name = data.get('model', 'linear')
    try:
        monthly_income = float(data.get('monthly_income', ''))
        avg_utility_bill = float(data.get('avg_utility_bill', ''))
        mobile_spend = float(data.get('mobile_spend', ''))
        savings_amount = float(data.get('savings_amount', ''))
        years_at_job = int(data.get('years_at_job', ''))
        utilization_rate = float(data.get('utilization_rate', ''))
        days_since_overdraft = int(data.get('days_since_overdraft', ''))
        is_entrepreneur = int(data.get('is_entrepreneur', ''))
        past_on_time_payments = int(data.get('past_on_time_payments', ''))
    except (ValueError, TypeError):
        return JsonResponse({'error': 'Invalid or missing input values.'}, status=400)

    import numpy as np
    from joblib import load
    if model_name == 'catboost':
        from catboost import CatBoostRegressor

        if not CATBOOST_MODEL_PATH.exists():
            return JsonResponse({'error': 'CatBoost model file not found.'}, status=404)
        model = CatBoostRegressor()
        model.load_model(str(CATBOOST_MODEL_PATH))
    else:
        if not LINEAR_MODEL_PATH.exists():
            return JsonResponse({'error': 'Linear model file not found.'}, status=404)
        model = load(LINEAR_MODEL_PATH)

    try:
        prediction = model.predict(
            np.array([[
                monthly_income,
                avg_utility_bill,
                mobile_spend,
                savings_amount,
                years_at_job,
                utilization_rate,
                days_since_overdraft,
                is_entrepreneur,
                past_on_time_payments,
            ]], dtype=float)
        )[0]
    except Exception as exc:
        return JsonResponse({'error': str(exc)}, status=500)

    return JsonResponse({
        'model': model_name,
        'monthly_income': monthly_income,
        'avg_utility_bill': avg_utility_bill,
        'mobile_spend': mobile_spend,
        'savings_amount': savings_amount,
        'years_at_job': years_at_job,
        'utilization_rate': utilization_rate,
        'days_since_overdraft': days_since_overdraft,
        'is_entrepreneur': is_entrepreneur,
        'past_on_time_payments': past_on_time_payments,
        'prediction': round(float(prediction), 2),
    })


def add_data_point(request):
    form = CreditApplicantForm(request.POST or None)
    message = None

    if request.method == 'POST' and form.is_valid():
        form.save()
        message = 'Credit applicant added; model will retrain on the next cycle.'
        form = CreditApplicantForm()

    return render(request, 'add_data_point.html', {'form': form, 'message': message})


def import_csv(request):
    form = DatasetUploadForm(request.POST or None, request.FILES or None)
    message = None

    if request.method == 'POST' and form.is_valid():
        import pandas as pd

        csv_file = form.cleaned_data['csv_file']
        df = pd.read_csv(csv_file)
        applicants = []
        for row in df.to_dict(orient='records'):
            applicants.append(CreditApplicant(
                monthly_income=row['monthly_income'],
                avg_utility_bill=row['avg_utility_bill'],
                mobile_spend=row['mobile_spend'],
                savings_amount=row['savings_amount'],
                years_at_job=int(row['years_at_job']),
                utilization_rate=float(row['utilization_rate']),
                days_since_overdraft=int(row['days_since_overdraft']),
                is_entrepreneur=bool(int(row['is_entrepreneur'])),
                past_on_time_payments=int(row['past_on_time_payments']),
                credit_score=float(row['credit_score']),
            ))
        CreditApplicant.objects.bulk_create(applicants)
        message = f'Imported {len(applicants)} rows successfully.'
        form = DatasetUploadForm()

    return render(request, 'import_csv.html', {'form': form, 'message': message})


def dashboard(request):
    import numpy as np
    from joblib import load
    from catboost import CatBoostRegressor

    applicants = list(CreditApplicant.objects.order_by('monthly_income'))
    actual_points = []

    for applicant in applicants:
        actual_points.append({
            'x': float(applicant.monthly_income),
            'y': float(applicant.credit_score),
            'label': (
                f"Income: {applicant.monthly_income}, "
                f"Savings: {applicant.savings_amount}, "
                f"Years at job: {applicant.years_at_job}, "
                f"Utilization: {applicant.utilization_rate}, "
                f"On-time payments: {applicant.past_on_time_payments}, "
                f"Score: {applicant.credit_score}"
            ),
        })

    linear_line = []
    catboost_line = []
    status = SystemStatus.objects.order_by('-updated_at').first()
    if applicants:
        incomes = np.array([float(a.monthly_income) for a in applicants])
        income_range = np.linspace(incomes.min(), incomes.max(), 100)
        median_values = {
            'avg_utility_bill': float(np.median([float(a.avg_utility_bill) for a in applicants])),
            'mobile_spend': float(np.median([float(a.mobile_spend) for a in applicants])),
            'savings_amount': float(np.median([float(a.savings_amount) for a in applicants])),
            'years_at_job': int(np.median([a.years_at_job for a in applicants])),
            'utilization_rate': float(np.median([a.utilization_rate for a in applicants])),
            'days_since_overdraft': int(np.median([a.days_since_overdraft for a in applicants])),
            'is_entrepreneur': int(np.median([int(a.is_entrepreneur) for a in applicants])),
            'past_on_time_payments': int(np.median([a.past_on_time_payments for a in applicants])),
        }

        line_features = np.column_stack([
            income_range,
            np.full_like(income_range, median_values['avg_utility_bill'], dtype=float),
            np.full_like(income_range, median_values['mobile_spend'], dtype=float),
            np.full_like(income_range, median_values['savings_amount'], dtype=float),
            np.full_like(income_range, median_values['years_at_job'], dtype=float),
            np.full_like(income_range, median_values['utilization_rate'], dtype=float),
            np.full_like(income_range, median_values['days_since_overdraft'], dtype=float),
            np.full_like(income_range, median_values['is_entrepreneur'], dtype=float),
            np.full_like(income_range, median_values['past_on_time_payments'], dtype=float),
        ])

        if LINEAR_MODEL_PATH.exists():
            linear_model = load(LINEAR_MODEL_PATH)
            linear_preds = linear_model.predict(line_features)
            linear_line = [
                {'x': float(x), 'y': float(y)} for x, y in zip(income_range, linear_preds)
            ]

        if CATBOOST_MODEL_PATH.exists():
            from catboost import CatBoostRegressor

            catboost_model = CatBoostRegressor()
            catboost_model.load_model(str(CATBOOST_MODEL_PATH))
            catboost_preds = catboost_model.predict(line_features)
            catboost_line = [
                {'x': float(x), 'y': float(y)} for x, y in zip(income_range, catboost_preds)
            ]

    return render(request, 'dashboard.html', {
        'actual_points': actual_points,
        'linear_line': linear_line,
        'catboost_line': catboost_line,
        'status': status,
    })


def upload_dataset(request):
    form = DatasetUploadForm(request.POST or None, request.FILES or None)
    message = None

    if request.method == 'POST' and form.is_valid():
        dataset = TrainingDataset(file=form.cleaned_data['csv_file'])
        dataset.save()
        message = 'Dataset uploaded successfully. The scheduler will retrain automatically.'
        return redirect('upload_dataset')

    recent_datasets = TrainingDataset.objects.order_by('-uploaded_at')[:10]
    recent_runs = TrainingRun.objects.order_by('-started_at')[:10]

    return render(
        request,
        'upload_dataset.html',
        {
            'form': form,
            'message': message,
            'datasets': recent_datasets,
            'runs': recent_runs,
        },
    )


def retrain_now(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required.'}, status=405)

    try:
        from .ml_logic import train_model_task

        result = train_model_task()
    except Exception as exc:
        return JsonResponse({'error': str(exc)}, status=500)

    return JsonResponse({
        'message': 'Retrained successfully.',
        'last_retrained': result.get('last_retrained'),
        'linear_r2': result['linear_metrics']['r2'],
        'catboost_r2': result['catboost_metrics']['r2'],
    })

def index(request):
    return render(request, 'prediction/index.html', {})


def predict_view(request):
    if request.method == 'POST':
        try:
            data = {
                'monthly_income_etb':        float(request.POST.get('monthly_income_etb', 0)),
                'telebirr_tx_count_30d':     int(request.POST.get('telebirr_tx_count_30d', 0)),
                'telebirr_avg_tx_value_etb': float(request.POST.get('telebirr_avg_tx_value_etb', 0)),
                'iqub_participation':        request.POST.get('iqub_participation') == 'true',
                'iqub_months_active':        int(request.POST.get('iqub_months_active', 0)),
                'merchant_category':         request.POST.get('merchant_category', 'retail'),
                'region':                    request.POST.get('region', 'addis_ababa_other'),
                'utility_bill_paid_on_time': request.POST.get('utility_bill_paid_on_time') == 'true',
                'mobile_topup_count_30d':    int(request.POST.get('mobile_topup_count_30d', 0)),
                'income_stability_score':    float(request.POST.get('income_stability_score', 0.5)),
                'business_segment':          request.POST.get('business_segment', 'shop'),
            }
            result = score_borrower(data)
            CreditApplication.objects.create(
                monthly_income_etb=data['monthly_income_etb'],
                telebirr_tx_count_30d=data['telebirr_tx_count_30d'],
                merchant_category=data['merchant_category'],
                region=data['region'],
                iqub_participation=data['iqub_participation'],
                credit_score=result['credit_score'],
                risk_tier=result['risk_tier'],
                approval_recommendation=result['approval_recommendation'],
                repayment_probability=result['repayment_probability'],
            )
            return render(request, 'prediction/index.html', {'result': result, 'input': data})
        except Exception as e:
            return render(request, 'prediction/index.html', {'error': str(e)})
    return render(request, 'prediction/index.html', {})