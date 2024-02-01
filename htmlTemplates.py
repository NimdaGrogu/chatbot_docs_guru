css = '''
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #2b313e
}
.chat-message.bot {
    background-color: #475063
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 78px;
  max-height: 78px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
}
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="1077115.png" style="max-height: 78px; max-width: 78px; border-radius: 50%; object-fit: cover;">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''


user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://i.ibb.co/R4sqwXd/1077114.jpg" style="max-height: 78px; max-width: 78px; border-radius: 50%; object-fit: cover;">>
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''
markdown = """
    <style>
    .stApp {
        background-color: #f0f0f0;
    }
    .stTextInput {
        padding: 10px;
        border: 2px solid #ccc;
        border-radius: 10px;
        font-size: 16px;
    }
    .stButton {
        background-color: #0078d4;
        color: white;
        padding: 8px 15px;
        border: none;
        border-radius: 10px;
        font-size: 16px;
        cursor: pointer;
    }
    .stButton:hover {
        background-color: #005a9e;
    }
    .chat-container {
        max-width: 70%;
        background-color: #0078d4;
        padding: 10px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
    }
    </style>
    """
