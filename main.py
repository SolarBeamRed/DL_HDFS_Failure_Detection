from src.models.train import train_model
from src.models.evaluate import evaluate_model
from src.models.summary import summarise_model
from src.models.predict_from_user import detect_anomalies
from rich.console import Console
from rich.traceback import install
from pyfiglet import Figlet
install()
console = Console()

def print_menu():
    console.rule(characters='=', style='#1000bb')
    console.print(Figlet(font='small', width=100).renderText("HDFS LOG ANOMALY DETECTION"), style="#2be9fb bold", justify='center')
    console.rule(characters='=', style='#1000bb')
    console.print('[underline]Available options[/]:', style="#ff6000 italic", justify='left')
    print('\n')
    console.print('1. Build, train and save model', style='#00ff02', justify='left')
    console.print('2. Evaluate model on test set', style='#00ff02', justify='left')
    console.print('3. Obtain model summary', style='#00ff02', justify='left')
    console.print('4. Upload Log file and perform operations', style='#00ff02', justify='left')
    console.print('5. Exit', style='#00ff02', justify='left')
    console.rule(characters='-', style='#1000bb')

def get_choice():
    try:
        return int(input('Enter your choice: ').strip())
    except ValueError:
        return None

def main():

    while True:
        print_menu()
        ch = get_choice()

        if ch == 1:
            console.rule(characters='-', style='#1000bb')
            console.print(Figlet(font='small', width=100).renderText('[TRAIN MODEL]'), style="bold green", justify='center')
            train_model()
            console.rule(characters='-', style='#1000bb')

        elif ch == 2:
            console.rule(characters='-', style='#1000bb')
            console.print(Figlet(font='small', width=100).renderText('[EVALUATE MODEL]'), style='bold yellow', justify='center')
            evaluate_model()
            console.rule(characters='-', style='#1000bb')

        elif ch == 3:
            console.rule(characters='-', style='#1000bb')
            console.print(Figlet(font='small', width=100).renderText('[MODEL SUMMARY]'),style="bold #ff0082", justify='center')
            summarise_model()
            console.rule(characters='-', style='#1000bb')

        elif ch == 4:
            console.rule(characters='-', style='#1000bb')
            console.print(Figlet(font='small', width=100).renderText('[LOG FILE INFERENCE]'), style="bold #ff0082",
                          justify='center')
            detect_anomalies()
            console.rule(characters='-', style='#1000bb')

        elif ch == 5:
            console.print('Exiting, thanks for using!', style='green italic')
            break

        else:
            console.print('Input not valid, please try again!', style='red italic bold')
            continue

        console.print('\n\n')

if __name__ == '__main__':
    main()
