from flask_wtf import FlaskForm
from wtforms import IntegerField, SubmitField, StringField
from wtforms.validators import ValidationError, DataRequired

class imgIdForm(FlaskForm):
	imgid=IntegerField("imageID",validators=[DataRequired()])
	submit=SubmitField("Search!")

class textForm(FlaskForm):
	textfield=StringField('captions',validators=[DataRequired()])
	submit=SubmitField('Search!')
	