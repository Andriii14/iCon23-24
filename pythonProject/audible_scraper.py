from selenium.webdriver.common.action_chains import ActionChains
import pandas as pd
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
import re

def web_driver():
    options = webdriver.ChromeOptions()
    options.add_argument("--verbose")
    options.add_argument('--no-sandbox') # needed, because colab runs as root
    options.add_argument('--disable-gpu')
    options.add_argument("--window-size=1920, 1200")
    options.add_argument('--disable-dev-shm-usage')
    # options.add_argument('--headless')  # Headless Chrome
    driver = webdriver.Chrome(options=options)
    driver.implicitly_wait(2)  # Small implicit wait
    return driver

def navigate_to_next_page(driver, wait, last_page_no, page):
  try:
    # Scroll to bottom of the page (might be needed for infinite scroll)
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

    # Try using different methods to click the "next" button
    try:
      next_button = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR,
                                                        '.bc-button.bc-button-secondary.nextButton.refinementFormButton.bc-button-small.bc-button-inline')))
      next_button.click()
    except:
      print('Clicking next button with element.click() failed.')
      try:
        # Use JavaScript click
        next_button = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR,
                                                            '.bc-button.bc-button-secondary.nextButton.refinementFormButton.bc-button-small.bc-button-inline')))
        driver.execute_script("arguments[0].click();", next_button)
      except:
        print('Clicking next button with JavaScript failed.')
        try:
          # Use ActionChains click
          next_button = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR,
                                                            '.bc-button.bc-button-secondary.nextButton.refinementFormButton.bc-button-small.bc-button-inline')))
          action = ActionChains(driver)
          action.click(next_button).perform()
        except:
          print('Clicking next button with ActionChains failed.')
          exit("Button to next page not found.")

    print('Moving to next page.')

  except TimeoutException:
    print('Next button not found or page load timed out.')

def extract_audiobook_info(driver):
    audiobook_data = {}

    # Try finding the title element with a wait (handles dynamic content)
    wait = WebDriverWait(driver, 10)  # Wait for 10 seconds at most
    try:
        # Find title element
        title_element = driver.find_element(By.CSS_SELECTOR, 'h1')
        audiobook_data['title'] = title_element.text
        # Check for and add subtitle (optional)
        try:
            subtitle_element = driver.find_element(By.CSS_SELECTOR, 'li.bc-list-item.subtitle.bc-spacing-s2.bc-size-medium')
            audiobook_data['subtitle'] = subtitle_element.text
            # audiobook_data['title'] += " - " + audiobook_data['subtitle']  # Combine title and subtitle
        except:
            audiobook_data['subtitle'] = None
    except:
        print("Title element not found after waiting.")
        audiobook_data['title'] = None  # Set to None if timed out

    try:
        # Find element with series attribute
        series_element = driver.find_element(By.CSS_SELECTOR, 'li.bc-list-item.seriesLabel')
        series_text = series_element.text.strip()
        if series_text.startswith("Serie: "):
            series_text = series_text[7:]
        audiobook_data['series'] = series_text
    except:
        audiobook_data['series'] = None # Set to None if element not found

    # Extract author (assuming element exists)
    try:
        author_element = driver.find_element(By.CSS_SELECTOR,
                                             'li.bc-list-item.authorLabel')  # li.bc-list-item.authorLabel a
        author_text = author_element.text.strip()
        if author_text.startswith("Di: "):
            author_text = author_text[4:]
        audiobook_data['author'] = author_text
    except:
        audiobook_data['author'] = None  # Set to None if element not found

    # Extract narrator (assuming element exists)
    try:
        narrator_element = driver.find_element(By.CSS_SELECTOR, 'li.bc-list-item.narratorLabel')
        audiobook_data['narrator'] = narrator_element.text.strip()[
                                     10:] if narrator_element and narrator_element.text.startswith(
            'Letto da: ') else None
    except:
        audiobook_data['narrator'] = None

    try:
        # Find duration element
        duration_element = driver.find_element(By.CSS_SELECTOR, 'li.bc-list-item.runtimeLabel')
        audiobook_data['duration'] = duration_element.text.strip()[
                                     8:] if duration_element and duration_element.text.startswith(
            'Durata: ') else None
    except:
        audiobook_data['duration'] = None  # Set to None if not found

    try:
        # Find rating element (replace with your selector)
        rating_element = driver.find_element(By.CSS_SELECTOR, 'li.bc-list-item.ratingsLabel')
        rating_text = rating_element.text.strip()  # Remove leading/trailing whitespace and newline

        # Extract rating value and parenthesized info (remove newline)
        rating_value, parenthesized_info = rating_text.split(' out of ', 1)

        # Extract review count (handle presence/absence of "5 stars")
        if '5 stars' in parenthesized_info:
            review_count = parenthesized_info.split('5 stars')[1].strip()
        else:
            review_count = parenthesized_info.strip()

        # Extract numerical value from review count (remove all non-digit characters)
        review_count = re.sub(r"\D", "", review_count)  # Use regular expression to remove all non-digit characters

        review_count = review_count[2:]  # Slice the string to remove the first two characters

        # Convert to integer and store values
        audiobook_data['rating'] = rating_value.strip().replace('\n', '')
        audiobook_data['review_count'] = int(review_count)
    except:
        audiobook_data['rating'] = None  # Set to None if not found
        audiobook_data['review_count'] = None  # Set to None if not found

    # Find all summary elements within the section
    try:
        summary_elements = driver.find_elements(By.CSS_SELECTOR, 'div.bc-box.bc-box-padding-none.bc-spacing-s2')
        # Initialize empty string for summaries
        audiobook_data['summary'] = ""

        # Extract, clean, and concatenate summaries
        for element in summary_elements:
            summary_text = element.text.strip().replace('\n', ' ')  # Replace newline with space
            audiobook_data['summary'] += summary_text + " "  # Add newline for separation
    except:
        audiobook_data['summary'] = None  # Set to None if no elements found

    try:
        # Find element with category attribute
        category_element = driver.find_element(By.CSS_SELECTOR, 'a.bc-link.navigation-link.bc-size-base.bc-color-link[aria-level="0"]')
        # Extract category attribute value
        audiobook_data['category'] = category_element.text
    except:
        audiobook_data['category'] = None # Set to None if element not found

    try:
        # Find element with subcategory attribute
        subcategory_element = driver.find_element(By.CSS_SELECTOR, 'a.bc-link.navigation-link.bc-size-base.bc-color-link[aria-level="1"]')
        # Extract subcategory attribute value
        audiobook_data['subcategory'] = subcategory_element.text
    except:
        audiobook_data['subcategory'] = None # Set to None if element not found

    # Find tags element
    try:
        tags_element = driver.find_element(By.CSS_SELECTOR, 'div.bc-expander-content')
        # Extract tags and remove newline character
        audiobook_data['tags'] = tags_element.text.strip().replace('\n', ', ')
    except:
        audiobook_data['tags'] = None  # Set to None if element not found

    # Extract additional information (duration, release date, category)
    # ... (use similar logic with try-except blocks)

    audiobook_data['link'] = driver.current_url
    audiobook_data['podcast_type'] = False

    # Find publisher element
    try:
        publisher_element = driver.find_element(By.CSS_SELECTOR, 'li.bc-list-item.publisherLabel')
        # Extract publisher and remove extra character
        audiobook_data['publisher'] = publisher_element.text.strip().replace('Editore: ', '')
    except:
        audiobook_data['publisher'] = None  # Set to None if element not found

    try:
        # Estrai le recensioni
        audiobook_data.update(extract_reviews(driver))  # Unisci i dizionari
    except NoSuchElementException:
        print("Reviews not found")
        # Inizializza le colonne delle recensioni con valori vuoti
        for i in range(1, 11):  # Crea 10 colonne vuote per le recensioni (puoi modificare il numero)
            audiobook_data[f'review_{i}'] = None

    return audiobook_data

def extract_podcast_info(driver):
    # Extract podcast title
    try:
        title_element = driver.find_element(By.CSS_SELECTOR, 'h1')
        podcast_title = title_element.text.strip()
    except NoSuchElementException:
        podcast_title = None

    # Extract podcast author
    try:
        author_element = driver.find_element(By.CSS_SELECTOR,
                                             '.bc-col-responsive.bc-col-5 > span > ul > li:nth-child(2)')
        podcast_author = author_element.text.strip()
        if podcast_author.startswith("Di: "):
            podcast_author = podcast_author[4:]
    except NoSuchElementException:
        print("Author element not found.")
        podcast_author = None

    try:
        # Extract narrator (assuming element exists)
        narrator_element = driver.find_element(By.CSS_SELECTOR, 'li.bc-list-item.narratorLabel')
        podcast_narrator = narrator_element.text.strip()[
                           10:] if narrator_element and narrator_element.text.startswith(
            'Letto da: ') else None
    except:
        podcast_narrator = None  # Set to None if not found

    try:
        category_element = driver.find_element(By.CSS_SELECTOR,
                                               'a.bc-link.navigation-link.bc-size-base.bc-color-link[aria-level="0"]')
        podcast_category = category_element.text.strip()
    except:
        podcast_category = None  # Set to None if not found

    try:
        subcategory_element = driver.find_element(By.CSS_SELECTOR, 'a.bc-link.navigation-link.bc-size-base.bc-color-link[aria-level="1"]')
        podcast_subcategory = subcategory_element.text.strip()
    except:
        podcast_subcategory = None  # Set to None if not found

    try:
        series_element = driver.find_element(By.CSS_SELECTOR, 'a.bc-link.navigation-link.bc-size-base.bc-color-link[aria-level="2"]')
        podcast_series = series_element.text.strip()
    except:
        podcast_series = None  # Set to None if not found

    try:
        # Find rating element (replace with your selector)
        rating_element = driver.find_element(By.CSS_SELECTOR, 'li.bc-list-item.ratingsLabel')
        rating_text = rating_element.text.strip()  # Remove leading/trailing whitespace and newline

        # Extract rating value and parenthesized info (remove newline)
        rating_value, parenthesized_info = rating_text.split(' out of ', 1)

        # Extract review count (handle presence/absence of "5 stars")
        if '5 stars' in parenthesized_info:
            review_count = parenthesized_info.split('5 stars')[1].strip()
        else:
            review_count = parenthesized_info.strip()

        # Extract numerical value from review count (remove all non-digit characters)
        review_count = re.sub(r"\D", "", review_count)  # Use regular expression to remove all non-digit characters

        review_count = review_count[2:]  # Slice the string to remove the first two characters

        # Convert to integer and store values
        podcast_rating = rating_value.strip().replace('\n', '')
        podcast_review = int(review_count)

    except:
        podcast_rating = None  # Set to None if not found
        podcast_review = None  # Set to None if not found

    try:
        # Find duration element (check if the css selector works)
        duration_element = driver.find_element(By.CSS_SELECTOR, 'li.bc-list-item.runtimeLabel')
        podcast_duration = duration_element.text.strip()[
                                     8:] if duration_element and duration_element.text.startswith(
            'Durata: ') else None
    except:
        podcast_duration = None  # Set to None if not found

    # Extract podcast summary
    try:
        description_element = driver.find_element(By.CSS_SELECTOR, '.bc-box.bc-box-padding-none.bc-spacing-s2 p')
        podcast_summary = description_element.text.strip().replace('\n', ' ')  # Extract and clean text content
    except NoSuchElementException:
        podcast_summary = None  # Set to None if not found

    try:
        tags_element = driver.find_element(By.CSS_SELECTOR, 'div.bc-section.bc-chip-group')
        # Extract tags and remove newline character
        podcast_tags = tags_element.text.strip().replace('\n', ', ')
    except:
        podcast_tags = None  # Set to None if not found

    current_url = driver.current_url
    podcast_boolean = True

    # Create and return podcast information dictionary
    podcast_info = {
        "title": podcast_title,
        "series": podcast_series,
        "author": podcast_author,
        "narrator": podcast_narrator,
        "duration": podcast_duration,
        "category": podcast_category,  # Add category with hardcoded value
        "subcategory": podcast_subcategory,
        "rating": podcast_rating,
        "review_count": podcast_review,
        "summary": podcast_summary,
        "tags": podcast_tags,
        "link": current_url,  # Add link to the dictionary
        "podcast_type": podcast_boolean
    }

    try:
        # Estrai le recensioni
        podcast_info.update(extract_reviews(driver))  # Unisci i dizionari
    except NoSuchElementException:
        print("Reviews not found")
        # Inizializza le colonne delle recensioni con valori vuoti
        for i in range(1, 11):  # Crea 10 colonne vuote per le recensioni (puoi modificare il numero)
            podcast_info[f'review_{i}'] = None

    return podcast_info

def extract_reviews(driver, max_reviews=10):
    reviews = {}
    review_elements = driver.find_elements(By.CSS_SELECTOR, 'div.bc-col-responsive.ITreviews0.bc-col-9')

    for i, review_element in enumerate(review_elements[:max_reviews]):  # Limita il numero di recensioni
        review_text = review_element.find_element(By.CSS_SELECTOR, '.bc-text.bc-spacing-small.bc-spacing-top-none.bc-size-body.bc-color-secondary').text.strip().replace('\n', '')
        reviews[f'review_{i + 1}'] = review_text

    return reviews

def identify_content_type(driver):
    try:
        # Check for element and extract text (if it exists)
        element = driver.find_element(By.CSS_SELECTOR, 'li.bc-list-item.format')
        element_text = element.text.strip()  # Extract and strip text content
    except NoSuchElementException:
        element_text = ""  # Set to empty string if element not found

    # Check for presence of "Podcast" (case-insensitive)
    if 'Podcast' in element_text:
        return "podcast"

    # If no clear indicators are found, assume it's an audiobook
    return "audiobook"

# per qualche motivo se non sei loggato la subcateogoria non viene mostrata
def login_to_audible(driver):
    # Replace these placeholders with your actual credentials
    email = 'email@email.com'
    password = 'password'

    # Navigate to login page
    driver.get('https://www.audible.it/sign-in')
    time.sleep(3)

    # Locate email input field and enter email
    email_field = driver.find_element(By.ID, 'ap_email')
    email_field.send_keys(email)

    # Locate and click "Continua" button
    continue_button = driver.find_element(By.ID, 'continue')  # Adjust ID if needed
    continue_button.click()

    # Wait for password field to become visible
    password_field = WebDriverWait(driver, 10).until(
        EC.visibility_of_element_located((By.ID, 'ap_password'))
    )
    password_field.send_keys(password)

    # Locate and click "Accedi" button
    login_button = driver.find_element(By.ID, 'signInSubmit')
    login_button.click()

    # Attesa fissa di X secondi (Adjust as needed)
    print("Attesa di 3 secondi per la conferma del login via email...")
    time.sleep(3)


def main():
  start_time = time.time() #checking time execution

  base_url = 'https://www.audible.it/search?feature_six_browse-bin=21876427031&sort=review-rank&ref_pageloadid=c0FTNPK9cVjk9pQU&ref=a_search_l1_catBackAll&pf_rd_p=950f1d06-0be7-4dc3-a8f5-37eb1e9ecc50&pf_rd_r=PTDR0ZKPCCW6E59JYX8W&pageLoadId=wQDJQj9Imr3k52iU&creativeId=89c16a62-4cd5-487a-bc38-68e0434b056b'
  driver = web_driver()
  login_to_audible(driver)  # Perform login before starting the scraping process
  driver.get(base_url)

  # Trova tutte le categorie principali
  category_list = WebDriverWait(driver, 10).until(
      EC.presence_of_element_located((By.CSS_SELECTOR, 'ul.bc-list.bc-spacing-medium.bc-list-nostyle'))
  )

  # Inizia dal primo filtro (indice 0)
  current_filter_index = 0

  # Create empty list to store audiobook data
  all_audiobooks_data = []

  while True:  # Continua a ciclare fino a quando tutte le categorie sono state elaborate
      # Trova nuovamente gli elementi della categoria ad ogni iterazione
      category_items = category_list.find_elements(By.CSS_SELECTOR, 'a.bc-link.refinementFormLink.bc-color-base ') # li.bc-list-item.bc-spacing-mini

      if current_filter_index >= len(category_items):
          break  # Esci dal ciclo se tutte le categorie sono state elaborate

      # Scorri fino alla categoria corrente per renderla visibile
      category_item = category_items[current_filter_index]
      driver.execute_script("arguments[0].scrollIntoView();", category_item)
      time.sleep(1)  # Attesa breve per permettere lo scorrimento

      # Stampa il nome della categoria selezionata
      category_name = category_item.text
      print(f"Raccolta dati per la categoria: {category_name}")

      # Clicca sulla categoria corrente (usa JavaScript per evitare problemi di visibilità)
      driver.execute_script("arguments[0].click();", category_item)
      time.sleep(2)  # Attesa per il caricamento della pagina

      # Get the number of last page to scrape
      last_page = driver.find_elements(By.CSS_SELECTOR, '.bc-link.refinementFormLink.pageNumberElement.bc-color-link')
      last_page_no = int(
          last_page[-1].text) if last_page else 1  # Se non ci sono altre pagine, considerala come ultima pagina
      print('Numero di pagine:', last_page_no)

      # Loop through all pages
      for page in range(1, last_page_no + 1):
          print('Page', page)
          box = driver.find_elements(By.CSS_SELECTOR,
                                     '.bc-col-responsive.bc-spacing-top-none.bc-col-8 > div > div.bc-col-responsive.bc-col-6 > div > div > span > ul')

          # Process only a limited number of audiobooks per page (optional)
          for item in box[:30]:  # Adjust limit as needed
              link = item.find_element(By.CSS_SELECTOR, 'h3 a').get_attribute('href')

              # Open link in the current tab
              driver.get(link)
              time.sleep(2)  # Short wait to allow page load (adjust as needed)

              # Identify content type (audiobook or podcast)
              content_type = identify_content_type(driver)

              # Extract information based on content type
              if content_type == "audiobook":
                  audiobook_data = extract_audiobook_info(driver)
                  all_audiobooks_data.append(audiobook_data)
              elif content_type == "podcast":
                  audiobook_data = extract_podcast_info(driver)
                  all_audiobooks_data.append(audiobook_data)

              # Go back to the main page (previous URL in history)
              driver.back()

          # Call the function to navigate to the next page
          wait = WebDriverWait(driver, 5)  # Wait for 5 seconds at most
          if page != last_page_no:
              navigate_to_next_page(driver, wait, last_page_no, page)
          else:
              print('This is the last page.')

      # Torna alla pagina iniziale e passa al prossimo filtro
      driver.get(base_url)

      # Attendi che la pagina iniziale si ricarichi e trovi nuovamente la lista delle categorie
      category_list = WebDriverWait(driver, 10).until(
          EC.presence_of_element_located((By.CSS_SELECTOR, 'ul.bc-list.bc-spacing-medium.bc-list-nostyle'))
      )
      current_filter_index += 1

  # Convert extracted data to pandas DataFrame
  df = pd.DataFrame(all_audiobooks_data)

  # Save DataFrame to CSV file
  df.to_csv('audible_italiano_uncleaned.csv', index=False)

  driver.quit()

  end_time = time.time()

  total_time = end_time - start_time
  print(f"Total execution time: {total_time} seconds")

if __name__ == "__main__":
  main() # circa 50 minuti per estrarre 500 righe