from django import forms
from django.contrib.auth.models import User
from django.core.exceptions import ValidationError


class SignupForm(forms.ModelForm):
    password = forms.CharField(widget=forms.PasswordInput)

    class Meta:
        model = User
        fields = ['first_name', 'username', 'email', 'password']

    def clean(self):
        cleaned_data = super().clean()
        username = cleaned_data.get("username")
        first_name = cleaned_data.get("first_name")

        # Check for unique combination of username and first name
        if User.objects.filter(username=username, first_name=first_name).exists():
            raise ValidationError(
                "The combination of username and first name must be unique.")

        return cleaned_data
