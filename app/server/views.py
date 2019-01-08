import csv
import json
from io import TextIOWrapper
import itertools as it
import logging
from functools import reduce

from django.urls import reverse
from django.http import HttpResponse, HttpResponseRedirect
from django.shortcuts import get_object_or_404
from django.views import View
from django.views.generic import TemplateView, CreateView
from django.views.generic.list import ListView
from django.contrib.auth.mixins import LoginRequiredMixin
from django.contrib import messages
from django.db import transaction

from .permissions import SuperUserMixin
from .forms import ProjectForm
from .models import Document, Project, DocumentAnnotation, SequenceAnnotation, Seq2seqAnnotation, Annotation, Label
from app import settings

logger = logging.getLogger(__name__)

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

    class ImportFileError(Exception):
        def __init__(self, message):
            self.message = message
    
    class LabelNotCreatedError(Exception):
        def __init__(self, label_text):
            self.message = "Label '%s' is not created! Please, create it manually and redo the import." % label_text

    def extract_metadata_csv(self, row, text_col, header_without_text):
        vals_without_text = [val for i,val in enumerate(row) if i != text_col]
        return json.dumps(dict(zip(header_without_text, vals_without_text)))

    def csv_to_documents(self, project, file, text_key='text', id_key='id'):
        form_data = TextIOWrapper(file, encoding='utf-8')
        reader = csv.reader(form_data)
        
        maybe_header = next(reader)
        if maybe_header:
            if text_key in maybe_header:
                text_col = maybe_header.index(text_key)
            elif len(maybe_header) == 1:
                reader = it.chain([maybe_header], reader)
                text_col = 0
            else:
                raise DataUpload.ImportFileError("CSV file must have either a title with \"text\" column or have only one column ")

            header_without_text = [title for i,title in enumerate(maybe_header) 
                                   if i != text_col]

            if id_key in maybe_header:
                id_col = maybe_header.index(id_key)
            else:
                id_col = None

            return (
                Document(
                    id=row[id_col] if id_col is not None else None,
                    text=row[text_col], 
                    metadata=self.extract_metadata_csv(row, text_col, header_without_text),
                    project=project
                )
                for row in reader
            )
        else:
            return []

    def extract_metadata_json(self, entry, known_keys):
        copy = entry.copy()
        for key in known_keys:
            del copy[key]
        return json.dumps(copy)

    def json_to_annotations_for_doc_classification(self, user, document, entry, labels):
        labels_text = entry.get('labels', [])

        def f(label_text):
            if label_text in labels:
                return DocumentAnnotation(user=user, document=document, label=labels[label_text])
            else:
                raise DataUpload.LabelNotCreatedError(label_text)

        return list(map(f, labels_text))

    def json_to_annotations_for_seq_labeling(self, user, document, entry, labels):
        entities = entry.get('entities', [])

        def f(entity):
            (start,end,label_text) = entity
            if label_text in labels:
                return SequenceAnnotation(user=user, document=document, start_offset=start, end_offset=end, label=labels[label_text])
            else:
                raise DataUpload.LabelNotCreatedError(label_text)
                
        return list(map(f, entities))


    def json_to_annotations_for_seq2seq(self, user, document, entry): 
        sentences = entry.get('sentences', [])
        return [
            Seq2seqAnnotation(user=user, document=document, text=sentence)
            for sentence in sentences
        ]

    def json_to_annotations(self, project, user, document, entry, labels):
        if project.is_type_of(Project.DOCUMENT_CLASSIFICATION):
            return self.json_to_annotations_for_doc_classification(user, document, entry, labels)
        elif project.is_type_of(Project.SEQUENCE_LABELING):
            return self.json_to_annotations_for_seq_labeling(user, document, entry, labels)
        elif project.is_type_of(Project.Seq2seq):
            return self.json_to_annotations_for_seq2seq(user, document, entry)

    def json_to_document(self, project, user, entry, text_key, labels):
        known_keys = {text_key}
        document = Document(text=entry[text_key], metadata=self.extract_metadata_json(entry, known_keys), project=project)
        return (
            document,
            self.json_to_annotations(project, user, document, entry, labels)
        )

    def json_to_documents(self, project, user, file, text_key='text'):     
        parsed_entries = (json.loads(line) for line in file)
        labels = {label.text: label for label in Label.objects.filter(project=project)}
        
        return (
            self.json_to_document(project, user, entry, text_key, labels)
            for entry in parsed_entries
        )

    def bulk_create_annotations(self, project, annotations, batch_size):
        if project.is_type_of(Project.DOCUMENT_CLASSIFICATION):
            cls = DocumentAnnotation
        elif project.is_type_of(Project.SEQUENCE_LABELING):
            cls = SequenceAnnotation
        elif project.is_type_of(Project.Seq2seq):
            cls = Seq2seqAnnotation
        
        cls.objects.bulk_create(annotations, batch_size=batch_size)
        

    def post(self, request, *args, **kwargs):
        project = get_object_or_404(Project, pk=kwargs.get('project_id'))
        user = self.request.user
        import_format = request.POST['format']
        try:
            file = request.FILES['file'].file
            documents = []
            if import_format == 'csv':
                documents = self.csv_to_documents(project, file)
                
            elif import_format == 'json':
                documents = self.json_to_documents(project, user, file)

            batch_size = settings.IMPORT_BATCH_SIZE

            with transaction.atomic():
                while True:
                    batch = list(it.islice(documents, batch_size))
                    if not batch:
                        break

                    docs, annotations = zip(*batch)
                    annotations = reduce(lambda acc,anns: acc + anns, annotations, [])

                    Document.objects.bulk_create(docs, batch_size=batch_size)
                    self.bulk_create_annotations(project, annotations, batch_size=batch_size)

            return HttpResponseRedirect(reverse('dataset', args=[project.id]))
        except (DataUpload.ImportFileError, DataUpload.LabelNotCreatedError) as e:
            messages.add_message(request, messages.ERROR, e.message)
            return HttpResponseRedirect(reverse('upload', args=[project.id]))
        except Exception as e:
            logger.exception(e)
            messages.add_message(request, messages.ERROR, 'Something went wrong')
            return HttpResponseRedirect(reverse('upload', args=[project.id]))


class DataDownload(SuperUserMixin, LoginRequiredMixin, TemplateView):
    template_name = 'admin/dataset_download.html'


class DataDownloadFile(SuperUserMixin, LoginRequiredMixin, View):

    def get(self, request, *args, **kwargs):
        project_id = self.kwargs['project_id']
        project = get_object_or_404(Project, pk=project_id)
        docs = project.get_documents(is_null=False).distinct()
        export_format = request.GET.get('format')
        filename = '_'.join(project.name.lower().split())
        try:
            if export_format == 'csv':
                response = self.get_csv(filename, docs)
            elif export_format == 'json':
                response = self.get_json(filename, docs)
            return response
        except Exception as e:
            logger.exception(e)
            messages.add_message(request, messages.ERROR, "Something went wrong")
            return HttpResponseRedirect(reverse('download', args=[project.id]))

    def get_csv(self, filename, docs):
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename="{}.csv"'.format(filename)
        writer = csv.writer(response)
        for d in docs:
            writer.writerows(d.to_csv())
        return response

    def get_json(self, filename, docs):
        response = HttpResponse(content_type='text/json')
        response['Content-Disposition'] = 'attachment; filename="{}.json"'.format(filename)
        for d in docs:
            dump = json.dumps(d.to_json(), ensure_ascii=False)
            response.write(dump + '\n') # write each json object end with a newline
        return response


class DemoTextClassification(TemplateView):
    template_name = 'demo/demo_text_classification.html'


class DemoNamedEntityRecognition(TemplateView):
    template_name = 'demo/demo_named_entity.html'


class DemoTranslation(TemplateView):
    template_name = 'demo/demo_translation.html'
