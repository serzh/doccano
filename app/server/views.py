import csv
import json
from io import TextIOWrapper

from django.urls import reverse
from django.http import HttpResponse, HttpResponseRedirect
from django.shortcuts import get_object_or_404
from django.views import View
from django.views.generic import TemplateView, CreateView
from django.views.generic.list import ListView
from django.contrib.auth.mixins import LoginRequiredMixin

from .permissions import SuperUserMixin
from .forms import ProjectForm
from .models import Document, Project


class IndexView(TemplateView):
    template_name = 'index.html'


class ProjectView(LoginRequiredMixin, TemplateView):

    def get_template_names(self):
        project = get_object_or_404(Project, pk=self.kwargs['project_id'])
        return [project.get_template_name()]


class ProjectsView(LoginRequiredMixin, CreateView):
    form_class = ProjectForm
    template_name = 'projects.html'


class DatasetView(SuperUserMixin, LoginRequiredMixin, ListView):
    template_name = 'admin/dataset.html'
    paginate_by = 5

    def get_queryset(self):
        project = get_object_or_404(Project, pk=self.kwargs['project_id'])
        return project.documents.all()


class LabelView(SuperUserMixin, LoginRequiredMixin, TemplateView):
    template_name = 'admin/label.html'


class StatsView(SuperUserMixin, LoginRequiredMixin, TemplateView):
    template_name = 'admin/stats.html'


class GuidelineView(SuperUserMixin, LoginRequiredMixin, TemplateView):
    template_name = 'admin/guideline.html'


class DataUpload(SuperUserMixin, LoginRequiredMixin, TemplateView):
    template_name = 'admin/dataset_upload.html'

    def post(self, request, *args, **kwargs):
        project = get_object_or_404(Project, pk=kwargs.get('project_id'))
        import_format = request.POST['format']
        try:
            if import_format == 'csv':
                form_data = TextIOWrapper(
                    request.FILES['file'].file, encoding='utf-8')
                reader = csv.reader(form_data)
                titles = next(reader)

                try:
                    text_col = titles.index('text')
                except ValueError:
                    # return error more gracefully
                    raise Exception("No text title in the csv header!")

                titles_without_text = [title for i,title in enumerate(titles) if i != text_col]
                def extract_meta(line):
                    vals_without_text = [val for i,val in enumerate(line) if i != text_col]
                    return json.dumps(dict(zip(titles_without_text, vals_without_text)))

                Document.objects.bulk_create([
                    Document(text=line[text_col], meta=extract_meta(line), project=project)
                    for line in reader
                ])

            elif import_format == 'json':
                form_data = request.FILES['file'].file

                text_key = 'text'

                def extract_meta(entry):
                    json_map = json.loads(entry)
                    meta = json_map.copy()
                    del meta[text_key]
                    return json.dumps(meta)
                
                parsed_entries = [json.loads(entry) for entry in form_data]

                Document.objects.bulk_create([
                    Document(text=entry['text'], meta=extract_meta(entry), project=project)
                    for entry in parsed_entries
                ])
            return HttpResponseRedirect(reverse('dataset', args=[project.id]))
        except:
            return HttpResponseRedirect(reverse('upload', args=[project.id]))


class DataDownload(SuperUserMixin, LoginRequiredMixin, View):

    def get(self, request, *args, **kwargs):
        project_id = self.kwargs['project_id']
        project = get_object_or_404(Project, pk=project_id)
        filename = '_'.join(project.name.lower().split())
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename="{}.csv"'.format(filename)

        writer = csv.writer(response)
        rows = project.export_annotated_docs_to_csv()
        writer.writerows(rows)

        return response


class DemoTextClassification(TemplateView):
    template_name = 'demo/demo_text_classification.html'


class DemoNamedEntityRecognition(TemplateView):
    template_name = 'demo/demo_named_entity.html'


class DemoTranslation(TemplateView):
    template_name = 'demo/demo_translation.html'
